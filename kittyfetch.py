#!/usr/bin/env python3
"""kittyfetch - a cat lives in your idle terminal, next to a fastfetch pane and a tty-clock.

    python3 kittyfetch.py                  truecolor half-block pixels (default)
    python3 kittyfetch.py --ascii          plain ASCII, no colour
    python3 kittyfetch.py --coat siamese   pick the cat (default: random)

Keys:  q quit   l laser   y toss the yarn   c next cat   i fastfetch pane   a ascii/colour
Move the mouse over the terminal and a laser dot follows it.

Stdlib only (Python 3.8+, Linux/macOS). The cat is a side-view procedural rig:
spring-driven pose parameters, a gait clock for the feet, 2-bone IK legs, a
lagging spring tail, a spine that bends when it turns, and behaviours written as
generators so they can pause, chain and be interrupted.

Layout of this file: helpers, canvas, world (sky pane, clock), props (laser,
yarn, mug), the cat (rig, render, behaviours), fastfetch info, simulation and
text layer, terminal output, entry points.
"""
import argparse
import math
import os
import random
import select
import struct
import sys
import time
import zlib

TAU = math.tau
PI = math.pi


# ------------------------------------------------------------------ helpers
def clamp(x, a, b):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mixc(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def mulc(c, k):
    return (c[0] * k, c[1] * k, c[2] * k)


def lit(c, L):
    return (c[0] * L[0], c[1] * L[1], c[2] * L[2])


def _c8(v):
    return 255 if v > 255 else 0 if v < 0 else int(v)


def pack(c):
    return (_c8(c[0]) << 16) | (_c8(c[1]) << 8) | _c8(c[2])


def lum(p):
    return 0.299 * (p >> 16) + 0.587 * ((p >> 8) & 255) + 0.114 * (p & 255)


_NR = random.Random(99)
_NZ = [_NR.uniform(-1, 1) for _ in range(512)]


def noise(x):
    i = math.floor(x)
    f = x - i
    i &= 511
    a = _NZ[i]
    return a + (_NZ[(i + 1) & 511] - a) * f * f * (3 - 2 * f)


def spring(x, v, target, k, dt):
    """Critically damped spring, implicit step (stable for any dt)."""
    f = 1.0 + 2.0 * dt * k
    hkk = dt * k * k
    det = 1.0 / (f + dt * hkk)
    return (f * x + dt * v + dt * hkk * target) * det, (v + hkk * (target - x)) * det


def ik(ax, ay, tx, ty, l1, l2):
    """2-bone IK in the cat's side plane; the joint bends backwards (cat elbow/hock)."""
    dx, dy = tx - ax, ty - ay
    d = math.hypot(dx, dy)
    dmax = l1 + l2 - 1e-3
    if d > dmax:
        k = dmax / d
        dx, dy, d = dx * k, dy * k, dmax
        tx, ty = ax + dx, ay + dy
    d = max(d, abs(l1 - l2) + 1e-3)
    a = math.atan2(dy, dx) - math.acos(clamp((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1))
    return ax + l1 * math.cos(a), ay + l1 * math.sin(a), tx, ty


# ------------------------------------------------------------------ canvas
class Canvas:
    """Packed-RGB pixel buffer with antialiased circles/capsules."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.px = [0] * (w * h)
        self.clip = (0, 0, w, h)

    def circle(self, cx, cy, r, col, shade=0.0, a=1.0):
        if r <= 0.05:
            return
        X0, Y0, X1, Y1 = self.clip
        w, px = self.w, self.px
        ro, ri = r + 0.5, r - 0.5
        R, G, B = col
        sqrt = math.sqrt
        rr, gg, bb = _c8(R), _c8(G), _c8(B)
        pk = (rr << 16) | (gg << 8) | bb
        for y in range(max(Y0, int(cy - ro)), min(Y1, int(cy + ro) + 1)):
            dy = y + 0.5 - cy
            if dy * dy >= ro * ro:
                continue
            if shade:
                k = 1.0 - shade * dy / r
                rr, gg, bb = _c8(R * k), _c8(G * k), _c8(B * k)
                pk = (rr << 16) | (gg << 8) | bb
            xo = sqrt(ro * ro - dy * dy)
            xa, xb = max(X0, int(cx - xo)), min(X1, int(cx + xo) + 1)
            row = y * w
            ia = ib = xa
            if a >= 1.0 and ri > 0 and dy * dy < ri * ri:
                xi = sqrt(ri * ri - dy * dy)
                ia, ib = max(xa, math.ceil(cx - xi - 0.5)), min(xb, math.floor(cx + xi - 0.5) + 1)
                if ib > ia:
                    px[row + ia:row + ib] = [pk] * (ib - ia)
                else:
                    ia = ib = xa
            for rng in (range(xa, ia), range(ib, xb)):
                for x in rng:
                    dx = x + 0.5 - cx
                    cov = (r + 0.5 - sqrt(dx * dx + dy * dy)) * a
                    if cov <= 0:
                        continue
                    i = row + x
                    if cov >= 1:
                        px[i] = pk
                    else:
                        c = px[i]
                        r0, g0, b0 = c >> 16, (c >> 8) & 255, c & 255
                        px[i] = ((int(r0 + (rr - r0) * cov) << 16) | (int(g0 + (gg - g0) * cov) << 8)
                                 | int(b0 + (bb - b0) * cov))

    def capsule(self, ax, ay, bx, by, ra, rb, col, a=1.0, shade=0.0):
        vx, vy = bx - ax, by - ay
        if a >= 1.0 and min(ra, rb) > 2.2:  # thick: chain of span-filled circles is cheaper
            n = int(math.hypot(vx, vy) / (min(ra, rb) * 0.35)) + 1
            for i in range(n + 1):
                t = i / n
                self.circle(ax + vx * t, ay + vy * t, ra + (rb - ra) * t, col, shade)
            return
        X0, Y0, X1, Y1 = self.clip
        rm = max(ra, rb) + 0.5
        x0, x1 = max(X0, int(min(ax, bx) - rm)), min(X1, int(max(ax, bx) + rm) + 1)
        y0, y1 = max(Y0, int(min(ay, by) - rm)), min(Y1, int(max(ay, by) + rm) + 1)
        L2 = vx * vx + vy * vy or 1e-9
        rr, gg, bb = _c8(col[0]), _c8(col[1]), _c8(col[2])
        pk = (rr << 16) | (gg << 8) | bb
        w, px, sqrt, dr = self.w, self.px, math.sqrt, rb - ra
        for y in range(y0, y1):
            py = y + 0.5 - ay
            row = y * w
            for x in range(x0, x1):
                qx = x + 0.5 - ax
                t = (qx * vx + py * vy) / L2
                t = 0.0 if t < 0 else 1.0 if t > 1 else t
                dx, dy = qx - vx * t, py - vy * t
                cov = (ra + dr * t + 0.5 - sqrt(dx * dx + dy * dy)) * a
                if cov <= 0:
                    continue
                i = row + x
                if cov >= 1:
                    px[i] = pk
                else:
                    c = px[i]
                    r0, g0, b0 = c >> 16, (c >> 8) & 255, c & 255
                    px[i] = ((int(r0 + (rr - r0) * cov) << 16) | (int(g0 + (gg - g0) * cov) << 8)
                             | int(b0 + (bb - b0) * cov))

    def rect(self, x0, y0, x1, y1, col, a=1.0):
        X0, Y0, X1, Y1 = self.clip
        xa, xb = max(X0, int(round(x0))), min(X1, int(round(x1)))
        ya, yb = max(Y0, int(round(y0))), min(Y1, int(round(y1)))
        if xb <= xa:
            return
        for y in range(ya, yb):
            self.span(y, xa, xb, col, a)

    def span(self, y, xa, xb, col, a=1.0):
        w, px = self.w, self.px
        row = y * w
        if a >= 1.0:
            px[row + xa:row + xb] = [pack(col)] * (xb - xa)
            return
        rr, gg, bb = _c8(col[0]), _c8(col[1]), _c8(col[2])
        for i in range(row + xa, row + xb):
            c = px[i]
            r0, g0, b0 = c >> 16, (c >> 8) & 255, c & 255
            px[i] = (int(r0 + (rr - r0) * a) << 16) | (int(g0 + (gg - g0) * a) << 8) | int(b0 + (bb - b0) * a)

    def poly(self, pts, col, a=1.0):
        X0, Y0, X1, Y1 = self.clip
        ys = [p[1] for p in pts]
        n = len(pts)
        for y in range(max(Y0, int(min(ys))), min(Y1, int(max(ys)) + 1)):
            yc = y + 0.5
            xs = []
            for i in range(n):
                (xa, ya), (xb, yb) = pts[i], pts[(i + 1) % n]
                if (ya <= yc < yb) or (yb <= yc < ya):
                    xs.append(xa + (yc - ya) * (xb - xa) / (yb - ya))
            xs.sort()
            for j in range(0, len(xs) - 1, 2):
                xa, xb = max(X0, int(round(xs[j]))), min(X1, int(round(xs[j + 1])))
                if xb > xa:
                    self.span(y, xa, xb, col, a)

    def dot(self, x, y, col, a=1.0):
        xi, yi = int(x), int(y)
        X0, Y0, X1, Y1 = self.clip
        if X0 <= xi < X1 and Y0 <= yi < Y1:
            self.span(yi, xi, xi + 1, col, a)


# ------------------------------------------------------------------ world
BG = (12, 13, 18)   # assumed terminal background: pixels of exactly this colour are left transparent
BG_PK = pack(BG)
FONT = {'0': ('111', '101', '101', '101', '111'), '1': ('010', '110', '010', '010', '111'),
        '2': ('111', '001', '111', '100', '111'), '3': ('111', '001', '111', '001', '111'),
        '4': ('101', '101', '111', '001', '001'), '5': ('111', '100', '111', '001', '111'),
        '6': ('111', '100', '111', '101', '111'), '7': ('111', '001', '001', '001', '001'),
        '8': ('111', '101', '111', '101', '111'), '9': ('111', '101', '111', '001', '111'),
        ':': ('0', '1', '0', '1', '0')}


# hour, sky top, sky horizon, light on things seen through the window
TOD = [
    (0.0, (8, 10, 28), (22, 26, 58), (0.40, 0.45, 0.70)),
    (4.8, (10, 12, 32), (30, 30, 66), (0.40, 0.45, 0.70)),
    (6.2, (70, 80, 140), (250, 160, 120), (0.78, 0.66, 0.66)),
    (7.8, (90, 150, 220), (200, 215, 230), (0.96, 0.93, 0.88)),
    (12.0, (70, 140, 225), (170, 205, 240), (1.0, 1.0, 1.0)),
    (17.0, (80, 135, 210), (220, 200, 180), (1.0, 0.95, 0.88)),
    (19.0, (70, 60, 120), (250, 130, 90), (0.86, 0.68, 0.62)),
    (20.6, (20, 22, 55), (60, 45, 90), (0.48, 0.48, 0.72)),
    (24.0, (8, 10, 28), (22, 26, 58), (0.40, 0.45, 0.70)),
]


def tod_at(h):
    for i in range(len(TOD) - 1):
        h0, a0, b0, l0 = TOD[i]
        h1, a1, b1, l1 = TOD[i + 1]
        if h0 <= h <= h1:
            t = smooth((h - h0) / (h1 - h0))
            return mixc(a0, a1, t), mixc(b0, b1, t), mixc(l0, l1, t)
    return TOD[0][1:]


class Bird:
    """A bird outside the window: flies past, sometimes lands on the ledge."""

    def __init__(self, world, rng):
        self.w, self.rng = world, rng
        self.state = 'gone'
        self.next = rng.uniform(30, 90)
        self.x = self.y = self.vx = self.vy = 0.0
        self.flap = 0.0
        self.dir = 1
        self.perch_x = None
        self.t_end = 0.0
        self.peck = 0.0

    def update(self, dt, t):
        w, S, rng = self.w, self.w.S, self.rng
        gx0, gy0, gx1, gy1 = w.glass
        if self.state == 'gone':
            if t > self.next and w.day > 0.3 and w.weather != 'rain':
                self.dir = rng.choice((-1, 1))
                self.x = gx0 - 4 * S if self.dir > 0 else gx1 + 4 * S
                self.y = rng.uniform(gy0 + 3 * S, gy0 + (gy1 - gy0) * 0.5)
                self.vx, self.vy = self.dir * rng.uniform(10, 16) * S, 0.0
                self.perch_x = rng.uniform(gx0 + 6 * S, gx1 - 6 * S) if rng.random() < 0.6 else None
                self.state = 'fly'
            return
        self.flap += dt * 17
        if self.state == 'fly':
            if self.perch_x is not None:
                dist = (self.perch_x - self.x) * self.dir
                py = gy1 - 1.3 * S
                if dist < 14 * S:
                    self.vx = self.dir * max(1.5 * S, abs(self.vx) * math.exp(-3 * dt))
                    self.y += (py - self.y) * min(1, dt * 4)
                if dist < 0.4 * S:
                    self.state, self.t_end, self.y = 'perch', t + rng.uniform(6, 20), py
                    return
            else:
                self.vy -= 3 * S * dt
            self.x += self.vx * dt
            self.y += self.vy * dt + math.sin(t * 5) * 0.1 * S
            if not (gx0 - 8 * S < self.x < gx1 + 8 * S) or self.y < gy0 - 6 * S:
                self.state, self.next = 'gone', t + rng.uniform(40, 170)
        elif self.state == 'perch':
            self.peck = max(0.0, math.sin(t * 9)) * 0.4 * S if (t % 3) < 0.8 else 0.0
            if rng.random() < dt * 0.4:
                self.dir = -self.dir
            if t > self.t_end:
                self.spook()

    def spook(self):
        if self.state == 'perch':
            self.state, self.perch_x = 'fly', None
            self.vx, self.vy = self.dir * 14 * self.w.S, -7 * self.w.S

    def draw(self, cv, L):
        if self.state == 'gone':
            return
        S, d, x, y = self.w.S, self.dir, self.x, self.y
        body, belly = lit((70, 58, 50), L), lit((206, 110, 70), L)
        perched = self.state == 'perch'
        cv.capsule(x - 0.7 * S * d, y, x - 1.9 * S * d, y + (0.7 if perched else -0.2) * S, 0.35 * S, 0.2 * S, body)
        cv.circle(x, y, 0.9 * S, body)
        cv.circle(x + 0.2 * S * d, y + 0.3 * S, 0.55 * S, belly)
        hx, hy = x + 0.8 * S * d, y - 0.55 * S + self.peck
        cv.circle(hx, hy, 0.6 * S, body)
        cv.capsule(hx + 0.5 * S * d, hy, hx + 1.0 * S * d, hy + 0.1 * S, 0.22 * S, 0.1 * S, lit((230, 170, 60), L))
        if not perched:
            f = math.sin(self.flap)
            cv.capsule(x, y - 0.2 * S, x - 0.4 * S * d, y - 1.8 * S * f, 0.45 * S, 0.15 * S, body)


class World:
    """The terminal as a room: the status bar is the floor, the 'outside' pane's
    bottom border is a ledge, and a tty-clock sits above that pane."""

    def __init__(self, W, H, S, rng, ascii_mode, info_w=0):
        self.W, self.H, self.S, self.rng, self.ascii = W, H, S, rng, ascii_mode
        u = S
        R = H // 2
        self.gy = 2 * (R - 1)                    # top of the status bar: the floor
        self.floor_top = self.gy - 1.5 * u       # laser above this is "on the wall"
        pw = int(clamp(0.24 * W, 18, 44))
        self.pc0, self.pc1 = 2, 2 + pw           # outside pane border columns
        r1 = int(clamp(round((self.gy - 21 * u - 1) / 2), 5, R - 4))
        # left column: tty-clock on top, the outside pane under it
        self.ck = int(clamp((pw + 1) // 17, 0, 3))
        while self.ck and r1 - (2 + math.ceil(2.5 * self.ck) + 2) < 5:
            self.ck -= 1
        top = 2 + math.ceil(2.5 * self.ck) + 2 if self.ck else 1
        r0 = max(top, r1 - max(5, int(12 * u)))
        self.pr0, self.pr1 = r0, r1
        self.wx0, self.wx1 = self.pc0 + 1, self.pc1
        self.wy0, self.wy1 = 2 * (r0 + 1), 2 * r1
        self.sill_y = 2 * r1 + 1                 # cats stand on the pane's bottom border
        self.sx0, self.sx1 = float(self.pc0), float(self.pc1 + 1)
        self.glass = (self.wx0, self.wy0, self.wx1, self.wy1)
        self.bed_x, self.bed_hw = W - max(21 * u, 0.17 * W), 11 * u   # favourite nap spot
        if self.ck:
            self.clock_x, self.clock_y = (self.pc0 + self.pc1 + 1) / 2, 2 + 2.5 * self.ck
        else:
            self.clock_x, self.clock_y = (self.wx0 + self.wx1) / 2, (self.wy0 + self.wy1) / 2
        gx0, gy0, gx1, gy1 = self.glass
        self.stars = [(rng.uniform(gx0, gx1), rng.uniform(gy0, gy0 + (gy1 - gy0) * 0.7), rng.random() * 50)
                      for _ in range(max(6, (gx1 - gx0) * (gy1 - gy0) // int(30 * u * u + 1)))]
        self.roofs = []
        x = gx0 - rng.uniform(0, 5) * u
        while x < gx1:
            bw, bh = rng.uniform(5, 11) * u, rng.uniform(3, 8) * u
            wins = [(rng.uniform(x + u, x + bw - u), rng.uniform(gy1 - bh + u, gy1 - u), rng.random())
                    for _ in range(int(bw * bh / (u * u * 9)))]
            self.roofs.append((x, x + bw, gy1 - bh, wins))
            x += bw + rng.uniform(-1, 1.5) * u
        self.clouds = [[rng.uniform(gx0, gx1), rng.uniform(gy0 + 2 * u, gy0 + (gy1 - gy0) * 0.4),
                        rng.uniform(0.7, 1.3), rng.uniform(0.4, 1.1)] for _ in range(3)]
        self.rain = [[rng.uniform(gx0, gx1), rng.uniform(gy0, gy1)] for _ in range(int((gx1 - gx0) / u))]
        self.weather = rng.choice(['clear', 'clear', 'cloudy', 'rain'])
        self.weather_next = rng.uniform(400, 1200)
        self.bird = Bird(self, rng)
        self.hour, self.night, self.day = 12.0, 0.0, 1.0
        self.L, self.sky = (1, 1, 1), ((0, 0, 0), (0, 0, 0))
        self.bg = [0 if ascii_mode else BG_PK] * (W * H)

    def bounds(self, level):
        u = self.S
        if level == 'sill':
            lo, hi = self.sx0 + 8.5 * u, self.sx1 - 8.5 * u
            return (lo, hi) if lo < hi else ((lo + hi) / 2,) * 2
        return 14 * u, self.W - 15 * u

    def ground(self, level, x):
        return self.sill_y if level == 'sill' else self.gy

    def update(self, dt, t, hour):
        self.hour = hour
        top, bot, L = tod_at(hour)
        self.night = clamp((0.92 - L[0]) / 0.5, 0, 1)
        self.day = 1 - self.night
        if self.weather != 'clear':
            g = 0.5 if self.weather == 'rain' else 0.35
            grey = mulc((150, 155, 165), 0.25 + 0.75 * self.day)
            top, bot = mixc(top, grey, g), mixc(bot, grey, g)
            L = mulc(L, 0.86 if self.weather == 'rain' else 0.93)
        self.L, self.sky = ((1.0, 1.0, 1.0) if self.ascii else L), (top, bot)
        if t > self.weather_next:
            self.weather = self.rng.choice(['clear', 'clear', 'cloudy', 'rain'])
            self.weather_next = t + self.rng.uniform(400, 1500)
        gx0, gy0, gx1, gy1 = self.glass
        spd = 2.5 if self.weather == 'rain' else 1.0
        for c in self.clouds:
            c[0] += c[3] * self.S * dt * spd
            if c[0] - 6 * self.S * c[2] > gx1:
                c[0] = gx0 - 6 * self.S * c[2]
        if self.weather == 'rain':
            for r in self.rain:
                r[1] += 55 * self.S * dt
                r[0] += 8 * self.S * dt
                if r[1] > gy1:
                    r[1] -= gy1 - gy0
                    r[0] = self.rng.uniform(gx0 - 4 * self.S, gx1)
        self.bird.update(dt, t)

    def background(self):
        return self.bg

    def draw_back(self, cv, t):
        S, L = self.S, self.L
        gx0, gy0, gx1, gy1 = self.glass
        if gx1 > gx0 and gy1 > gy0:
            cv.clip = (gx0, gy0, gx1, gy1)
            top, bot = self.sky
            for y in range(gy0, gy1):
                cv.span(y, gx0, gx1, (0, 0, 0) if self.ascii else mixc(top, bot, (y - gy0) / max(1, gy1 - gy0 - 1)))
            star_k = self.night if not self.ascii else (1.0 if self.night > 0.5 else 0.0)
            if star_k > 0.05 and self.weather != 'rain':
                for sx, sy, ph in self.stars:
                    cv.dot(sx, sy, (255, 250, 230), star_k * (0.45 + 0.4 * noise(int(t * 4) / 4 * 0.7 + ph)))
            h = self.hour
            if 6 <= h <= 18:
                u, col, r = (h - 6) / 12, (255, 238, 180), 2.2 * S
            else:
                u, col, r = ((h - 18) % 24) / 12, (236, 236, 222), 1.8 * S
            bx = lerp(gx0 - 6 * S, gx1 + 6 * S, u)
            by = gy1 - (gy1 - gy0) * (0.25 + 0.6 * math.sin(PI * u))
            if self.weather == 'clear' or self.ascii:
                cv.circle(bx, by, r * 2.2, col, 0, 0.12)
                cv.circle(bx, by, r, col)
                if not 6 <= h <= 18:
                    cv.circle(bx + 0.8 * S, by - 0.4 * S, r * 0.85, mixc(top, bot, (by - gy0) / max(1, gy1 - gy0)))
            cc = (70, 70, 70) if self.ascii else \
                mixc(mixc(top, bot, 0.5), (250, 250, 255), 0.55 * (0.3 + 0.7 * self.day))
            if self.weather != 'clear':
                cc = mulc(cc, 0.85)
            for x, y, sc, _ in self.clouds[: (3 if self.weather != 'clear' else 2)]:
                x = round(x * 4) / 4
                for dx, dy, rr in ((0, 0, 2.2), (2.3, 0.5, 1.7), (-2.2, 0.6, 1.5), (0.9, -1.0, 1.6)):
                    cv.circle(x + dx * S * sc, y + dy * S * sc, rr * S * sc, cc, 0.15)
            if not self.ascii:
                roof = mulc(mixc(bot, (20, 22, 30), 0.65), 0.9)
                for x0, x1, ry, wins in self.roofs:
                    cv.rect(x0, ry, x1, gy1, roof)
                    if self.night > 0.3:
                        for wx, wy, ph in wins:
                            if ph < 0.45 + 0.2 * noise(t * 0.01 + ph * 40):
                                cv.dot(wx, wy, (255, 210, 120), self.night * 0.9)
            self.bird.draw(cv, L if not self.ascii else (2.2, 2.2, 2.2))
            if self.weather == 'rain':
                rc = (150, 160, 175) if not self.ascii else (110, 110, 110)
                for x, y in self.rain:
                    cv.capsule(x, y, x + 0.5 * S, y + 2.4 * S, 0.3, 0.3, rc, 0.55)
            cv.clip = (0, 0, cv.w, cv.h)
        # tty-clock, colon blinking with the seconds (ASCII mode draws it as text instead)
        if self.ck and not self.ascii:
            lt = time.localtime()
            k = self.ck
            txt = '%02d:%02d' % (lt.tm_hour, lt.tm_min)
            x = self.clock_x - 17 * k / 2
            col = (120, 120, 120) if self.ascii else (70, 80, 108)
            for ch in txt:
                for gy_, row in enumerate(FONT[ch]):
                    for gx_, bit in enumerate(row):
                        if bit == '1' and (ch != ':' or lt.tm_sec % 2 == 0):
                            cv.rect(x + gx_ * k, 2 + gy_ * k, x + (gx_ + 1) * k, 2 + (gy_ + 1) * k, col)
                x += (len(FONT[ch][0]) + 1) * k


# ------------------------------------------------------------------ props
class Laser:
    """The red dot: follows the mouse, or a pretend human waves it about."""

    def __init__(self, sim, rng):
        self.sim, self.rng = sim, rng
        self.on, self.mode = False, None
        self.x = self.y = 0.0
        self.next_session = rng.uniform(200, 500)
        self.session_end = 0.0
        self.mouse_t, self.mx, self.my = -99.0, 0.0, 0.0
        self.sx = self.sy = self.tx = self.ty = 0.0
        self.seg_t0, self.seg_dur, self.hold_until = 0.0, 1.0, 0.0

    def start(self, t, dur=None):
        w, S, cat = self.sim.world, self.sim.S, self.sim.cat
        self.on, self.mode = True, 'ai'
        self.session_end = t + (dur or self.rng.uniform(25, 70))
        self.x = clamp(cat.x + cat.dir * self.rng.uniform(25, 45) * S, 6 * S, w.W - 6 * S)
        self.y = w.gy - 0.8 * S
        self.new_seg(t)

    def stop(self, t):
        self.on, self.mode = False, None
        self.next_session = t + self.rng.uniform(300, 1100)

    def new_seg(self, t, fast=False):
        w, S, rng, cat = self.sim.world, self.sim.S, self.rng, self.sim.cat
        self.sx, self.sy = self.x, self.y
        r = rng.random()
        if fast:
            nx = self.x + rng.choice((-1, 1)) * rng.uniform(18, 40) * S
        elif r < 0.25:
            nx = cat.x + cat.dir * rng.uniform(16, 30) * S
        elif r < 0.75:
            nx = self.x + rng.uniform(-35, 35) * S
        else:
            nx = rng.uniform(8 * S, w.W - 8 * S)
        self.tx = clamp(nx, 6 * S, w.W - 6 * S)
        if not fast and rng.random() < 0.15:
            self.ty = rng.uniform(w.floor_top - 22 * S, w.floor_top - 6 * S)  # up the wall
        else:
            self.ty = w.gy - rng.uniform(0.6, 1.2) * S
        dist = math.hypot(self.tx - self.sx, self.ty - self.sy) / S
        self.seg_t0 = t
        self.seg_dur = 0.12 if fast else clamp(dist / rng.uniform(25, 90), 0.15, 1.6)
        self.hold_until = t + self.seg_dur + (0.05 if fast else rng.uniform(0.2, 2.2))

    def pounced(self, t):
        if self.mode == 'ai' and self.rng.random() < 0.7:
            self.seg_t0, self.hold_until = t + 0.06, t  # dodge just as the cat leaves the ground
            self.new_seg(t + 0.06, fast=True)

    def vanish(self, t):
        if self.mode == 'ai':
            S = self.sim.S
            self.x = clamp(self.x + self.rng.choice((-1, 1)) * self.rng.uniform(25, 40) * S, 6 * S, self.sim.W - 6 * S)
            self.sx, self.tx, self.sy, self.ty, self.hold_until = self.x, self.x, self.y, self.y, t + 1

    def update(self, dt, t):
        if t - self.mouse_t < 4.0:
            self.on, self.mode = True, 'mouse'
            k = min(1, dt * 25)
            self.x += (self.mx - self.x) * k
            self.y += (self.my - self.y) * k
            return
        if self.mode == 'mouse':
            self.on, self.mode = False, None
        if self.mode != 'ai':
            if t > self.next_session and not self.sim.cat.asleep:
                self.start(t)
            return
        if t > self.session_end:
            self.stop(t)
            return
        u = (t - self.seg_t0) / self.seg_dur
        if u < 0:
            return
        if u <= 1:
            e = smooth(u)
            self.x, self.y = lerp(self.sx, self.tx, e), lerp(self.sy, self.ty, e)
        elif t > self.hold_until:
            self.new_seg(t)
        else:
            self.x += noise(t * 9) * 0.15 * self.sim.S  # hand tremor

    def draw(self, cv, ascii_mode):
        if not self.on:
            return
        S = self.sim.S
        if ascii_mode:
            cv.circle(self.x, self.y, 0.9 * S, (255, 255, 255))
            return
        cv.circle(self.x, self.y, 2.0 * S, (255, 30, 40), 0, 0.25)
        cv.circle(self.x, self.y, 0.8 * S, (255, 70, 70))
        cv.circle(self.x, self.y, 0.35 * S + 0.3, (255, 210, 200))


class Yarn:
    """A ball of yarn that rolls, bounces and trails a loose strand."""

    def __init__(self, sim, rng):
        self.sim, self.rng = sim, rng
        S = sim.S
        self.r = 2.0 * S
        self.x = sim.W * 0.6
        self.y = self.ground()
        self.vx = self.vy = 0.0
        self.rot = 0.0
        self.air = False
        self.tossed = False
        self.strand = [(self.x - i * S, self.ground() + self.r) for i in range(6)]

    def ground(self):
        return self.sim.world.gy - self.r

    def kick(self, v, up=0.0):
        self.vx = v
        if up:
            self.vy, self.air = -up, True

    def toss(self):
        S, W = self.sim.S, self.sim.W
        self.x = self.rng.uniform(0.25, 0.75) * W
        self.y = -4 * S
        self.vx, self.vy, self.air = self.rng.uniform(-25, 25) * S, 0.0, True
        self.tossed = True

    def update(self, dt):
        S, W = self.sim.S, self.sim.W
        g = self.ground()
        if self.air:
            self.vy += 120 * S * dt
            self.y += self.vy * dt
            if self.y >= g:
                self.y = g
                if self.vy > 15 * S:
                    self.vy *= -0.4
                else:
                    self.vy, self.air = 0.0, False
        else:
            self.y = g
            self.vx *= math.exp(-1.3 * dt)
            if abs(self.vx) < 0.4 * S:
                self.vx = 0.0
        self.x += self.vx * dt
        lo, hi = 4 * S + self.r, W - 4 * S - self.r
        if not lo <= self.x <= hi:
            self.x = clamp(self.x, lo, hi)
            self.vx = -self.vx * 0.45
        self.rot += self.vx * dt / self.r
        # loose strand: rope that follows the ball and lies on the floor
        a = self.rot + 2.0
        pts = [(self.x + math.cos(a) * self.r, self.y + math.sin(a) * self.r)]
        seg = 1.2 * S
        floor = g + self.r - 0.3
        for i in range(1, len(self.strand)):
            px, py = self.strand[i]
            qx, qy = pts[-1]
            py = min(floor, py + 30 * S * dt)
            dx, dy = px - qx, py - qy
            d = math.hypot(dx, dy) or 1
            if d > seg:
                px, py = qx + dx / d * seg, qy + dy / d * seg
            pts.append((px, py))
        self.strand = pts

    def draw(self, cv, L, ascii_mode):
        S = self.sim.S
        base = (96, 156, 206) if not ascii_mode else (170, 170, 170)
        c = lit(base, L)
        dark = mulc(c, 0.7)
        for (ax, ay), (bx, by) in zip(self.strand, self.strand[1:]):
            cv.capsule(ax, ay, bx, by, 0.3 * S, 0.3 * S, c)
        cv.circle(self.x, self.y, self.r, c, 0.25)
        for k in range(3):
            a = self.rot + k * PI / 3
            ca, sa = math.cos(a), math.sin(a)
            cv.capsule(self.x - ca * self.r * 0.75, self.y - sa * self.r * 0.75, self.x + ca * self.r * 0.75,
                       self.y + sa * self.r * 0.75, 0.22 * S + 0.15, 0.22 * S + 0.15, dark)


class Mug:
    """Lives on the ledge. Gets knocked off. Is tidied back while the cat sleeps."""

    def __init__(self, sim, rng):
        self.sim, self.rng = sim, rng
        self.reset()

    def reset(self):
        w, S = self.sim.world, self.sim.S
        self.state = 'sill'
        self.hw, self.h = 1.3 * S, 2.9 * S
        self.x, self.y = w.sx1 - 3.2 * S, w.sill_y
        self.vx = self.vy = self.va = 0.0
        self.ang = 0.0
        self.alpha = 1.0
        self.t_floor = 0.0

    def push(self, dx):
        if self.state == 'sill':
            self.vx += dx * 7

    def update(self, dt, t):
        w, S = self.sim.world, self.sim.S
        if self.state == 'sill':
            self.x += self.vx * dt
            self.vx *= math.exp(-7 * dt)
            if self.x - self.hw * 0.2 > w.sx1:
                self.state = 'fall'
                self.vx = max(self.vx, 5 * S)
                self.va = self.rng.uniform(4, 8)
        elif self.state == 'fall':
            self.vy += 160 * S * dt
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.ang += self.va * dt
            fl = w.gy
            if self.y >= fl:
                self.y = fl
                if self.vy > 25 * S:
                    self.vy *= -0.3
                    self.va *= 0.5
                    self.sim.clink(self.x, self.y)
                else:
                    self.state, self.vy, self.t_floor = 'floor', 0.0, t
        elif self.state == 'floor':
            target = round(self.ang / (PI / 2)) * (PI / 2)
            self.ang += (target - self.ang) * min(1, dt * 8)
            self.x += self.vx * dt
            self.vx *= math.exp(-4 * dt)
            # someone tidies up while Miso sleeps
            if self.sim.cat.asleep and t - self.t_floor > 60:
                self.state = 'fade'
        elif self.state == 'fade':
            self.alpha -= dt * 0.5
            if self.alpha <= 0:
                self.reset()
                self.alpha, self.state = 0.0, 'appear'
        elif self.state == 'appear':
            self.alpha = min(1.0, self.alpha + dt * 0.5)
            if self.alpha >= 1:
                self.state = 'sill'

    def draw(self, cv, L, ascii_mode):
        if self.alpha <= 0.02:
            return
        S = self.sim.S
        ca, sa = math.cos(self.ang), math.sin(self.ang)
        cx, cy = self.x, self.y - self.h / 2

        def P(lx, ly):  # local: x right, y up from mug centre
            return cx + lx * ca + ly * sa, cy - ly * ca + lx * sa

        hw, hh = self.hw, self.h / 2
        body = lit((238, 234, 224), L) if not ascii_mode else (220, 220, 220)
        band = lit((70, 150, 170), L) if not ascii_mode else (120, 120, 120)
        a = self.alpha
        handle = (((hw, hh * 0.5), (hw + 0.8 * S, hh * 0.2)), ((hw + 0.8 * S, hh * 0.2), (hw, -hh * 0.5)))
        for (x0, y0), (x1, y1) in handle:
            A, B = P(x0, y0), P(x1, y1)
            cv.capsule(A[0], A[1], B[0], B[1], 0.3 * S, 0.3 * S, body, a)
        cv.poly([P(-hw, -hh), P(hw, -hh), P(hw, hh), P(-hw, hh)], body, a)
        cv.poly([P(-hw, -0.15 * hh), P(hw, -0.15 * hh), P(hw, 0.3 * hh), P(-hw, 0.3 * hh)], band, a)


# ------------------------------------------------------------------ the cat
POSE_BASE = dict(hx=-6.8, hy=8.4, sx=6.0, sy=9.0, arch=0.4, hdx=3.4, hdy=4.6, htilt=0.0, hyaw=0.0,
                 ffx=6.8, ffy=1.0, hfx=-7.2, hfy=1.0, ear=0.0, eye=1.0, pup=0.3, mouth=0.0,
                 tailA=1.0, tailC=0.14, tailK=9.0, tailZ=-0.6, tailS=0.12, puff=0.0, rb=1.0,
                 stride=0.0, gal=0.0)
POSES = {
    'stand': {},
    'sit': dict(hx=-4.2, hy=4.7, sx=1.6, sy=9.6, arch=0.3, hdx=1.7, hdy=5.6, ffx=4.0, hfx=-0.9,
                tailA=-1.0, tailC=-0.5, tailK=4.5, tailZ=2.2, tailS=0.04),
    'loaf': dict(hx=-5.4, hy=4.3, sx=3.8, sy=4.6, arch=0.9, hdx=3.0, hdy=4.3, ffx=4.8, ffy=1.3, hfx=-3.2,
                 hfy=1.3, eye=0.75, tailA=-0.9, tailC=-0.45, tailK=4.0, tailZ=1.3, tailS=0.03, rb=1.07),
    'curl': dict(hx=-4.2, hy=3.9, sx=2.8, sy=3.8, arch=1.4, hdx=3.6, hdy=-0.7, htilt=-0.3, ffx=5.2, ffy=1.1,
                 hfx=-2.4, hfy=1.1, eye=0.0, ear=0.3, tailA=-0.9, tailC=-0.55, tailK=5.0, tailZ=1.6,
                 tailS=0.015, rb=1.1),
    'crouch': dict(hx=-6.4, hy=5.6, sx=5.2, sy=5.4, arch=0.1, hdx=4.2, hdy=2.6, htilt=0.05, ffx=6.6, hfx=-6.4,
                   tailA=-0.1, tailC=0.02, tailS=0.04, pup=0.9),
    'stretch': dict(hx=-6.4, hy=9.6, sx=6.2, sy=3.6, arch=-1.6, hdx=4.6, hdy=1.4, htilt=0.35, ffx=13.0,
                    hfx=-6.4, ear=0.5, eye=0.15, tailA=1.5, tailC=0.04),
    'air': dict(hx=-7.0, hy=8.6, sx=6.6, sy=9.2, arch=-0.4, hdx=4.4, hdy=3.4, ffx=11.0, ffy=6.0, hfx=-12.5,
                hfy=5.5, tailA=0.35, tailC=0.05, ear=0.25),
    'reach': dict(hx=-3.6, hy=7.0, sx=0.8, sy=13.5, arch=-0.2, hdx=2.0, hdy=4.2, htilt=0.6, ffx=4.0, ffy=13.0,
                  hfx=-3.0, hfy=1.5, tailA=-0.3, tailC=0.0),
    'pin': dict(hx=-6.0, hy=7.2, sx=5.4, sy=5.2, arch=0.6, hdx=4.4, hdy=1.2, htilt=-0.45, ffx=8.6, hfx=-6.0,
                tailA=0.2, tailC=0.05),
    'arch': dict(hx=-4.6, hy=10.6, sx=4.0, sy=10.2, arch=4.6, hdx=3.0, hdy=2.6, htilt=-0.2, ffx=5.0, hfx=-5.0,
                 ear=0.8, pup=1.0, tailA=1.45, tailC=-0.02, puff=1.0),
    'skid': dict(hx=-6.6, hy=6.2, sx=5.0, sy=8.4, arch=0.0, hdx=3.6, hdy=4.0, htilt=0.2, ffx=10.0, hfx=-4.0,
                 ear=0.9, tailA=0.9, tailC=-0.1),
}
KSTIFF = dict(htilt=9, hyaw=7, ear=18, eye=26, mouth=22, pup=5, tailA=5, tailC=5, tailK=4, tailZ=6,
              tailS=3, puff=1.6, rb=4, stride=7, gal=2.5)
LEG_LAT = (1.5, -1.5, 1.7, -1.7)       # near-front, far-front, near-hind, far-hind
WALK_OFF = (0.25, 0.75, 0.0, 0.5)      # lateral-sequence walk
GAL_OFF = (0.55, 0.65, 0.0, 0.1)       # rotary gallop
TAIL_N, TAIL_SEG = 9, 1.75
TURN_K = 4.5                           # base turn stiffness (head x2.1, shoulders x1.45, hips x1)
# Coats. Missing keys fall back: paw/muzzle -> cream, face/ear/tail -> fur, ring/tip -> stripe, eye2 -> eye.
COATS = {
    'ginger': dict(name='Miso', desc='ginger tabby', fur=(232, 136, 58), stripe=(190, 94, 36), far=(184, 102, 46),
                   cream=(250, 234, 208), accent=(240, 150, 70)),
    'black': dict(name='Pepper', desc='black cat', fur=(64, 62, 72), stripe=(54, 52, 61), far=(46, 45, 52),
                  cream=(64, 62, 72), cream_f=(46, 45, 52), eye=(236, 190, 60), nose=(80, 66, 72),
                  pink=(122, 84, 96), whisk=(150, 150, 160), accent=(236, 190, 60)),
    'grey': dict(name='Luna', desc='silver tabby', fur=(156, 160, 170), stripe=(88, 92, 104), far=(120, 124, 134),
                 cream=(228, 230, 234), eye=(120, 196, 96), accent=(170, 180, 200)),
    'tuxedo': dict(name='Oreo', desc='tuxedo', fur=(62, 60, 68), stripe=(56, 54, 62), far=(46, 45, 52),
                   cream=(242, 242, 244), cream_f=(196, 196, 200), eye=(150, 210, 80), nose=(226, 140, 150),
                   accent=(220, 220, 230)),
    'siamese': dict(name='Mochi', desc='siamese', fur=(238, 226, 204), stripe=(228, 214, 188), far=(206, 192, 168),
                    cream=(246, 240, 226), cream_f=(208, 198, 180), paw=(98, 74, 60), paw_f=(80, 60, 50),
                    ear=(98, 74, 60), tail=(124, 96, 78), ring=(108, 82, 66), tip=(80, 60, 50),
                    muzzle=(112, 86, 70), eye=(110, 172, 242), nose=(70, 52, 44), pink=(150, 110, 96),
                    accent=(110, 172, 242)),
    'tortie': dict(name='Biscuit', desc='tortoiseshell', fur=(66, 50, 42), stripe=(212, 126, 58), far=(52, 40, 34),
                   cream=(90, 66, 52), cream_f=(64, 48, 40), ring=(196, 112, 50), eye=(226, 176, 60),
                   nose=(96, 70, 62), accent=(212, 126, 58)),
    'white': dict(name='Yuki', desc='odd-eyed white', fur=(240, 240, 238), stripe=(226, 226, 226),
                  far=(204, 204, 208), cream=(250, 250, 250), cream_f=(214, 214, 218), eye=(110, 170, 240),
                  eye2=(230, 180, 60), nose=(240, 160, 172), accent=(200, 210, 240)),
}
CAT_BASE = dict(cream_f=(206, 184, 158), pink=(238, 150, 156), nose=(214, 110, 118), eye=(178, 216, 70),
                pupil=(22, 22, 20), dark=(84, 44, 30), whisk=(250, 244, 230))


def coat_palette(name):
    c = dict(CAT_BASE)
    c.update(COATS[name])
    for k, src in (('paw', 'cream'), ('paw_f', 'cream_f'), ('muzzle', 'cream'), ('face', 'fur'), ('ear', 'fur'),
                   ('tail', 'fur'), ('ring', 'stripe'), ('tip', 'stripe'), ('eye2', 'eye')):
        c.setdefault(k, c[src])
    return c


class Cat:
    """Rig, renderer, drives and behaviours of the cat."""

    def __init__(self, sim, rng):
        self.sim, self.rng = sim, rng
        self.S = sim.S
        self.t = 0.0
        self.dt = 0.05
        self.tgt = dict(POSE_BASE)
        self.cur = dict(POSE_BASE)
        self.vel = {k: 0.0 for k in POSE_BASE}
        self.posture = 'stand'
        self.set_pose('sit')
        self.cur.update(self.tgt)
        w = sim.world
        self.level = 'floor'
        self.x = lerp(*w.bounds('floor'), 0.45)
        self.y = w.ground('floor', self.x)
        self.dir, self.th, self.thv, self.th_k = 1, 0.0, 0.0, TURN_K
        # heading cascade: head leads, shoulders follow, hips trail (self.th = shoulders)
        self.thh = self.thhv = self.thr = self.thrv = self.th_goal = 0.0
        self.turning, self.dip = False, 0.0
        self.v = self.vt = 0.0
        self.acc, self.dec = 12.0, 16.0
        self.phase = 0.0
        self.gx, self.gy = [0.0] * 4, [0.0] * 4
        self.bob_h = self.bob_s = self.flex = 0.0
        self.pex, self.pey, self.pexv, self.peyv = [0.0] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 4
        self.pext, self.peyt = [0.0] * 4, [0.0] * 4
        self.lift, self.liftv, self.fwx = [0.0] * 4, [0.0] * 4, [None] * 4
        self.ta = [PI - 1.0 - 0.14 * i for i in range(TAIL_N)]
        self.tv = [0.0] * TAIL_N
        self.tail_tw, self.tail_freq = 0.0, 0.3
        self.earT, self.earV = [0.0, 0.0], [0.0, 0.0]
        self.blink, self.blink_t, self.next_blink, self.next_ear = 1.0, -9.0, 2.0, 3.0
        self.breath, self.breath_amp = 0.0, 0.4
        self.look = None
        self.energy, self.boredom, self.mischief = 0.75, 0.3, 0.35
        self.asleep = False
        self.gen, self.gen_name, self.gen_t0 = None, '', 0.0
        self.history = []
        self.force = []
        self.busy = self.airborne = False
        self.snap = 1.0
        self.doing = 'sitting'
        self.hunt_cool = self.zoom_cool = 0.0
        self.seed = rng.random() * 100
        self.coat = sim.coat

    # ----------------------------------------------------------- pose plumbing
    def set_pose(self, name, **kw):
        keep = self.tgt['stride'], self.tgt['gal']
        self.tgt.update(POSE_BASE)
        self.tgt.update(POSES[name])
        self.tgt['stride'], self.tgt['gal'] = keep
        self.tgt.update(kw)
        self.posture = name

    def head_pos(self):
        c = self.cur
        return (self.x + (c['sx'] + c['hdx']) * math.cos(self.th) * self.S,
                self.y - (c['sy'] + c['hdy'] + 3) * self.S)

    def resized(self, sim, fx):
        self.sim, self.S = sim, sim.S
        w = sim.world
        if self.gen:
            self.gen.close()
            self.gen = None
        self.busy = self.airborne = False
        self.v = self.vt = 0.0
        self.x = clamp(self.x * fx, *w.bounds(self.level))
        self.y = w.ground(self.level, self.x)
        self.fwx = [None] * 4

    # ----------------------------------------------------------- frame update
    def update(self, dt):
        self.dt = dt
        self.t += dt
        self.think(dt)
        self.idle(dt)
        self.physics(dt)
        self.animate(dt)
        self.drives(dt)

    def think(self, dt):
        if not self.busy:
            want = self.interrupt(dt)
            if want:
                self.switch(want)
        if self.gen is None:
            self.switch(self.choose())
        try:
            next(self.gen)
        except StopIteration:
            self.gen = None

    def interrupt(self, dt):
        s, g = self.sim, self.gen_name
        if s.laser.on and g not in ('hunt', 'startle') and self.t > self.hunt_cool:
            if not self.asleep or self.rng.random() < dt * 0.4:
                return 'hunt'
        if s.startle_evt:
            s.startle_evt = False
            if g not in ('hunt', 'zoomies', 'startle'):
                return 'startle'
        if s.yarn.tossed:
            if not self.asleep and g not in ('hunt', 'zoomies', 'knock', 'yarn', 'startle'):
                s.yarn.tossed = False
                return 'yarn'
            if self.asleep and self.rng.random() < dt * 0.5:
                s.yarn.tossed = False
        if s.world.bird.state == 'perch' and not self.asleep and g in ('sit', 'loaf', 'groom', 'wander') \
                and self.rng.random() < dt * 0.25:
            return 'window'
        return None

    def choose(self):
        if self.force:
            return self.force.pop(0)
        s, e, b, m = self.sim, self.energy, self.boredom, self.mischief
        h = s.world.hour
        night, dusk = h < 5 or h > 22.5, 5 <= h < 7.5 or 18 <= h < 21
        zoom_ok = self.t > self.zoom_cool and e > 0.45
        w = {
            'nap': 0.1 + 2.4 * (1 - e) ** 2 + (0.8 if e < 0.3 else 0) + (0.3 if night else 0),
            'sit': 0.55,
            'loaf': 0.3 + 0.5 * (1 - e),
            'groom': 0.45,
            'wander': 0.35 + 0.6 * b,
            'window': 0.2 + 0.45 * s.world.day + (2.0 if s.world.bird.state != 'gone' else 0),
            'yarn': 0.05 + 1.3 * e * b,
            'zoomies': (0.03 + 0.9 * e * e * b) * (2.5 if night or dusk else 1) * zoom_ok,
            'knock': 1.8 * m * m * e if (s.mug.state == 'sill' and m > 0.45) else 0,
        }
        for i, name in enumerate(reversed(self.history[-3:])):
            if name in w:
                w[name] *= (0.12, 0.35, 0.6)[i]
        r = self.rng.random() * sum(w.values())
        for name, v in w.items():
            r -= v
            if r <= 0:
                return name
        return 'sit'

    def switch(self, name):
        if self.gen:
            self.gen.close()
        self.vt = 0.0
        self.look = None
        self.busy = False
        self.snap, self.th_k, self.tail_tw = 1.0, TURN_K, 0.0
        self.turning, self.dip = False, 0.0
        self.pext, self.peyt = [0.0] * 4, [0.0] * 4
        self.tgt['mouth'], self.tgt['stride'], self.tgt['pup'] = 0.0, 0.0, 0.3
        if self.asleep and name not in ('hunt', 'startle'):
            self.asleep = False
        self.gen_name, self.gen_t0 = name, self.t
        self.history = (self.history + [name])[-8:]
        self.gen = getattr(self, 'b_' + name)()

    def idle(self, dt):
        rng = self.rng
        if not self.asleep and self.t > self.next_blink:
            self.blink_t = self.t
            self.next_blink = self.t + (0.35 if rng.random() < 0.15 else rng.uniform(2.0, 6.5))
        bt = self.t - self.blink_t
        self.blink = 1.0 if bt > 0.16 else abs(bt - 0.08) / 0.08
        if self.t > self.next_ear:
            i = rng.randrange(2)
            self.earV[i] += rng.choice((-1, 1)) * rng.uniform(6, 12)
            self.next_ear = self.t + rng.uniform(2.5, 11)
        for i in (0, 1):
            self.earT[i], self.earV[i] = spring(self.earT[i], self.earV[i], 0.0, 14, dt)
        self.breath += dt * TAU * (0.27 if self.asleep else 0.45)
        self.breath_amp += ((1.0 if self.asleep else 0.4) - self.breath_amp) * min(1, dt)

    def physics(self, dt):
        w, S = self.sim.world, self.S
        if self.v < self.vt:
            self.v = min(self.vt, self.v + self.acc * dt)
        else:
            self.v = max(self.vt, self.v - self.dec * dt)
        if not self.airborne:
            lo, hi = w.bounds(self.level)
            self.x += self.v * self.dir * S * dt
            if not lo - 3 * S <= self.x <= hi + 3 * S:
                self.x = clamp(self.x, lo - 3 * S, hi + 3 * S)
                self.v = self.vt = 0.0
            gy = w.ground(self.level, self.x)
            self.y += (gy - self.y) * min(1, dt * 12)

    def look_targets(self):
        lk = self.look() if callable(self.look) else self.look
        if lk == 'viewer':
            return 0.05, 1.0
        S, c = self.S, self.cur
        hx = self.x + (c['sx'] + c['hdx']) * self.dir * S
        hy = self.y - (c['sy'] + c['hdy']) * S
        dxu, dyu = (lk[0] - hx) / S * self.dir, (hy - lk[1]) / S
        if len(lk) > 2 and lk[2]:  # through the window: turn the face away from us
            return clamp(math.atan2(dyu, 14), -0.6, 0.8), -1 + clamp(dxu / 18, -1, 1) * 0.75
        if dxu > -3:
            return clamp(math.atan2(dyu, max(dxu, 2)), -0.9, 0.9), 0.0
        return clamp(math.atan2(dyu, -dxu), -0.7, 0.8), 1.75

    def animate(self, dt):
        cur, vel = self.cur, self.vel
        eff = dict(self.tgt)
        if self.look is not None:
            eff['htilt'], eff['hyaw'] = self.look_targets()
        if self.energy < 0.3 and self.gen_name not in ('hunt', 'zoomies', 'startle'):
            eff['eye'] = min(eff['eye'], 0.72)
        snap = self.snap
        for k, v in eff.items():
            cur[k], vel[k] = spring(cur[k], vel[k], v, KSTIFF.get(k, 10) * snap, dt)
        k = self.th_k
        if not self.turning:  # while turning, face() drives the headings directly
            self.thh, self.thhv = spring(self.thh, self.thhv, self.th_goal, k * 2.1, dt)
            self.th, self.thv = spring(self.th, self.thv, self.th_goal, k * 1.45, dt)
            self.thr, self.thrv = spring(self.thr, self.thrv, self.th_goal, k * 1.0, dt)
        for i in range(4):
            self.pex[i], self.pexv[i] = spring(self.pex[i], self.pexv[i], self.pext[i], 16 * snap, dt)
            self.pey[i], self.peyv[i] = spring(self.pey[i], self.peyv[i], self.peyt[i], 16 * snap, dt)
        # gait clock: feet are a function of travelled distance, so they never skate
        gal, stride = cur['gal'], clamp(cur['stride'], 0, 1)
        L = 8.5 + 9.5 * gal
        duty = 0.62 - 0.24 * gal
        lift_h = 1.5 + 1.6 * gal
        self.phase = (self.phase + self.v * dt / L) % 1.0
        for i in range(4):
            ph = (self.phase + lerp(WALK_OFF[i], GAL_OFF[i], gal)) % 1.0
            if ph < duty:
                fx, fy = 0.5 - ph / duty, 0.0
            else:
                s = (ph - duty) / (1 - duty)
                fx, fy = -0.5 + smooth(s), math.sin(PI * s)
            self.gx[i] = fx * L * duty * stride
            self.gy[i] = fy * lift_h * stride
        P = self.phase * TAU
        self.bob_h = stride * ((1 - gal) * 0.3 * math.cos(2 * P) + gal * 1.1 * math.sin(P + 0.6))
        self.bob_s = stride * ((1 - gal) * 0.3 * math.cos(2 * P + 1.2) + gal * 1.1 * math.sin(P + 0.6 + PI))
        self.flex = stride * gal * 1.4 * math.sin(P)
        # feet that move sideways without the gait (turning, shuffling) step instead of sliding
        c = math.cos(self.th)
        for i in range(4):
            base = (cur['ffx'] if i < 2 else cur['hfx']) + self.gx[i] + self.pex[i]
            wx = self.x + base * c * self.S
            sp = abs(wx - self.fwx[i]) / dt / self.S if self.fwx[i] is not None and dt > 0 else 0.0
            self.fwx[i] = wx
            on_ground = (cur['ffy'] if i < 2 else cur['hfy']) < 1.6 and not self.airborne
            alt = max(0.0, math.sin(self.t * TAU * 2.2 + (0 if i in (0, 3) else PI)))
            tg = clamp(sp * 0.11, 0, 1.6) * (1 - stride) * alt if on_ground else 0.0
            self.lift[i], self.liftv[i] = spring(self.lift[i], self.liftv[i], tg, 22, dt)
        # tail: per-segment angle springs, stiffer at the root so the tip lags (overlap)
        base = PI - cur['tailA']
        cum = 0.0
        trail = min(0.7, self.v / 45)
        for i in range(TAIL_N):
            cum += cur['tailC'] * clamp(cur['tailK'] - i, 0, 1)
            sway = cur['tailS'] * math.sin(self.t * self.tail_freq * TAU - i * 0.55) * (0.3 + i / TAIL_N)
            tw = self.tail_tw * noise(self.t * 7 + i * 0.3) * (i / TAIL_N) ** 2 * 1.5
            tg = base - cum + sway + tw
            tg += (PI - tg) * trail * (0.3 + 0.7 * i / TAIL_N)
            self.ta[i], self.tv[i] = spring(self.ta[i], self.tv[i], tg, 15 - 8 * i / TAIL_N, dt)

    def drives(self, dt):
        if self.asleep:
            self.energy = min(1.0, self.energy + dt / 150)
            self.boredom = max(0.0, self.boredom - dt / 200)
        else:
            self.energy = max(0.0, self.energy - dt * (1 / 700 + self.v / 40 / 90))
            g = self.gen_name
            if g in ('sit', 'loaf', 'groom', 'window'):
                self.boredom = min(1.0, self.boredom + dt / 110)
            elif g in ('yarn', 'hunt', 'zoomies', 'knock'):
                self.boredom = max(0.0, self.boredom - dt / 25)
            else:
                self.boredom = min(1.0, self.boredom + dt / 300)
        self.mischief = min(1.0, self.mischief + dt / 500)

    def mood(self):
        if self.asleep:
            return 'dreaming' if self.breath_amp > 0.9 and math.sin(self.t * 0.05) > 0.6 else 'asleep'
        g = self.gen_name
        if g == 'zoomies':
            return 'unhinged'
        if g == 'startle':
            return 'spooked'
        if g == 'hunt':
            return 'predatory'
        if self.energy < 0.3:
            return 'sleepy'
        if self.mischief > 0.65:
            return 'scheming'
        if self.boredom > 0.6:
            return 'bored'
        return 'curious' if g in ('wander', 'window') else 'content'

    # ----------------------------------------------------------- render
    def palette(self, L):
        return {k: (lit(v, L) if isinstance(v, tuple) else v) for k, v in coat_palette(self.coat).items()}

    def draw(self, cv, L):
        S, p = self.S, self.cur
        thf, thr = self.th, self.thr
        X0, Y0 = self.x, self.y
        pal = self.palette(L)
        prims = []
        add = prims.append

        puff = p['puff']
        br = math.sin(self.breath) * self.breath_amp
        rb = p['rb'] * (1 + 0.035 * br) * (1 + 0.25 * puff)
        hx, hy = p['hx'] - self.flex, p['hy'] + self.bob_h + 0.18 * br
        sx, sy = p['sx'] + self.flex, p['sy'] + self.bob_s + 0.12 * br
        vx, vy = sx - hx, sy - hy
        vl = math.hypot(vx, vy) or 1.0
        vx, vy = vx / vl, vy / vl
        nx, ny = -vy, vx                      # "up" normal of the spine
        arch = p['arch']
        mx, my = (hx + sx) / 2 + nx * arch, (hy + sy) / 2 + ny * arch
        hipR, chR = 4.5 * rb, 4.15 * rb
        sy -= 1.2 * self.dip                  # the chest dips as the cat swings round

        def ang(f):  # heading blends from hips to shoulders
            span = sx - hx
            return thf if abs(span) < 1e-3 else lerp(thr, thf, smooth((f - hx) / span))

        # Lay the spine out by integrating that heading from the root, so mid-turn
        # the body is a real C/L-shaped arc instead of a rigid shape rotating.
        n, SP = 40, 0.5
        sxs, szs = [0.0] * (2 * n + 1), [0.0] * (2 * n + 1)
        for i in range(n + 1, 2 * n + 1):
            a = ang((i - n - 0.5) * SP)
            sxs[i], szs[i] = sxs[i - 1] + SP * math.cos(a), szs[i - 1] + SP * math.sin(a)
        for i in range(n - 1, -1, -1):
            a = ang((i - n + 0.5) * SP)
            sxs[i], szs[i] = sxs[i + 1] - SP * math.cos(a), szs[i + 1] - SP * math.sin(a)

        def spine(f):
            q = f / SP + n
            i = min(2 * n - 1, max(0, int(math.floor(q))))
            t = q - i
            return sxs[i] + (sxs[i + 1] - sxs[i]) * t, szs[i] + (szs[i + 1] - szs[i]) * t

        def W(f, u, l=0.0, a=None, anchor=None):
            f0 = f if anchor is None else anchor
            px_, _ = spine(f0)
            a = ang(f0) if a is None else a
            return X0 + (px_ + (f - f0) * math.cos(a) - l * math.sin(a)) * S, Y0 - u * S

        def D(f, l=0.0, a=None, anchor=None):
            f0 = f if anchor is None else anchor
            _, pz = spine(f0)
            a = ang(f0) if a is None else a
            return pz + (f - f0) * math.sin(a) + l * math.cos(a)

        def B(t):
            a, b, cc = (1 - t) ** 2, 2 * (1 - t) * t, t * t
            return a * hx + b * mx + cc * sx, a * hy + b * my + cc * sy

        # torso: a spine curve swept with circles, plus haunch and chest
        top = max(D(B(i / 7)[0]) for i in range(8))  # belly/stripes sit on the torso surface
        for i in range(8):
            t = i / 7
            f, u = B(t)
            r = lerp(hipR * 0.9, chR * 0.94, t) - math.sin(t * PI) * 0.3 * rb
            X, Y = W(f, u)
            add((D(f), 0, X, Y, r * S, pal['fur'], 0.14))
        X, Y = W(hx - 0.2, hy + 0.1, a=thr)
        add((D(hx, 0, thr) + 0.3, 0, X, Y, hipR * S, pal['fur'], 0.16))
        cf, cu = sx + vx * 0.5, sy + vy * 0.5
        X, Y = W(cf, cu, a=thf)
        add((D(cf, 0, thf) + 0.2, 0, X, Y, chR * S, pal['fur'], 0.14))
        for t in (0.35, 0.45, 0.55, 0.65, 0.75):  # cream belly
            f, u = B(t)
            X, Y = W(f - nx * 1.9 * rb, u - ny * 1.9 * rb)
            add((top + 0.05, 0, X, Y, 2.1 * rb * S, pal['cream'], 0.0))
        X, Y = W(sx + vx * 1.6 - nx * 1.6, sy + vy * 1.6 - ny * 1.6)
        add((D(sx) + 0.25, 0, X, Y, 2.3 * rb * S, pal['cream'], 0.0))  # bib
        for t in (0.25, 0.47, 0.69):  # tabby stripes over the back
            f, u = B(t)
            r = lerp(hipR * 0.9, chR * 0.94, t)
            A = W(f + nx * r * 0.95, u + ny * r * 0.95)
            Bp = W(f + nx * r * 0.35 - vx * 0.8, u + ny * r * 0.35 - vy * 0.8)
            add((top + 0.06, 1, A[0], A[1], Bp[0], Bp[1], 0.6 * S, 0.25 * S, pal['stripe'], 1.0))
        for a0, a1 in (((-2.4, 3.0), (-0.6, 1.4)), ((-3.6, 1.4), (-1.8, -0.4))):  # haunch stripes
            A, Bp = W(hx + a0[0], hy + a0[1], a=thr), W(hx + a1[0], hy + a1[1], a=thr)
            add((D(hx, 0, thr) + 0.31, 1, A[0], A[1], Bp[0], Bp[1], 0.5 * S, 0.22 * S, pal['stripe'], 1.0))

        # legs: authored base + gait clock + behaviour offsets, solved with IK
        for i in range(4):
            lat = LEG_LAT[i]
            if i < 2:
                af, au = sx + 0.5, sy - 1.6
                bf, bu = p['ffx'] - (0.5 if i == 1 else 0), p['ffy']
                l1, l2, r1a, r1b = 3.8, 3.6, 1.5, 1.15
            else:
                af, au = hx + 0.5, hy - 1.4
                bf, bu = p['hfx'] + (0.6 if i == 3 else 0), p['hfy']
                l1, l2, r1a, r1b = 3.4, 5.0, 2.2, 1.25
            ff = bf + self.gx[i] + self.pex[i]
            fu = max(1.0, bu + self.gy[i] + self.pey[i] + self.lift[i])
            jf, ju, ff, fu = ik(af, au, ff, fu, l1, l2)
            ju = max(ju, 0.9)
            a = thf if i < 2 else thr
            near = lat * math.cos(a) >= 0
            col = pal['fur'] if near else pal['far']
            paw = pal['paw'] if near else pal['paw_f']
            dep = D(af, lat, a)
            A, J, F, T = (W(af, au, lat, a), W(jf, ju, lat, a, af), W(ff, fu, lat, a, af),
                          W(ff + 0.8, fu - 0.05, lat, a, af))
            add((dep, 1, A[0], A[1], J[0], J[1], r1a * S, r1b * S, col, 1.0))
            add((dep + 0.001, 1, J[0], J[1], F[0], F[1], 1.05 * S, 1.0 * S, col, 1.0))
            add((dep + 0.002, 1, F[0], F[1], T[0], T[1], 1.08 * S, 1.0 * S, paw, 1.0))

        # tail
        tf, tu = hx - vx * hipR * 0.75, hy - vy * hipR * 0.75 + 0.6
        pts = [(tf, tu)]
        f_, u_ = tf, tu
        for a in self.ta:
            f_ += TAIL_SEG * math.cos(a)
            u_ = max(0.8, u_ + TAIL_SEG * math.sin(a))
            pts.append((f_, u_))
        tz = p['tailZ'] + D(hx, 0, thr) * 0.5
        for i in range(TAIL_N):
            r0 = lerp(1.2, 0.85, i / TAIL_N) * (1 + 1.3 * puff)
            r1 = lerp(1.2, 0.85, (i + 1) / TAIL_N) * (1 + 1.3 * puff)
            col = pal['tip'] if i == TAIL_N - 1 else pal['ring'] if i % 2 == 1 else pal['tail']
            A, Bp = W(*pts[i], a=thr, anchor=tf), W(*pts[i + 1], a=thr, anchor=tf)
            add((tz + i * 1e-4, 1, A[0], A[1], Bp[0], Bp[1], r0 * S, r1 * S, col, 1.0))

        # head: its own yaw (look at you / out of the window / behind) and tilt
        yaw = p['hyaw']
        HXl, HUl = sx + p['hdx'] * math.cos(yaw * PI / 2), sy + p['hdy']
        R = 3.9 * (1 + 0.1 * puff)
        HX, HY = W(HXl, HUl, a=thf, anchor=sx)
        thh = self.thh + (1 if math.cos(self.thh) >= 0 else -1) * yaw * PI / 2
        ch, sh = math.cos(thh), math.sin(thh)
        ct, st = math.cos(p['htilt']), math.sin(p['htilt'])
        hd = D(HXl, 0, thf, sx) + 0.9

        def HP(f, u, l=0.0):
            f2, u2 = f * ct - u * st, f * st + u * ct
            return HX + (f2 * ch - l * sh) * S, HY - u2 * S, f2 * sh + l * ch

        for ei, l in ((0, 0.42), (1, -0.42)):
            bx_, by_, bd = HP(-0.05 * R, 0.68 * R, l * R)
            e = p['ear'] + self.earT[ei]
            tx_, ty_, _ = HP((-0.15 - 0.8 * e) * R, (1.62 - 0.55 * e) * R, l * 1.3 * R)
            dep = hd + (0.3 if bd > 0 else -0.3)
            add((dep, 1, bx_, by_, tx_, ty_, 0.46 * R * S, 0.07 * R * S, pal['ear'], 1.0))
            if bd >= 0 and sh > -0.4:
                ix, iy, _ = HP(0.05 * R, 0.8 * R, l * R)
                jx, jy, _ = HP((-0.1 - 0.6 * e) * R, (1.35 - 0.45 * e) * R, l * 1.2 * R)
                add((dep + 0.001, 1, ix, iy, jx, jy, 0.22 * R * S, 0.05 * R * S, pal['pink'], 1.0))
        add((hd, 0, HX, HY, R * S, pal['face'], 0.12))
        X, Y, _ = HP(-0.1 * R, -0.38 * R)
        add((hd + 0.001, 0, X, Y, 0.8 * R * S, pal['face'], 0.05))
        mxx, myy, md = HP(0.62 * R, -0.33 * R)
        if md > -0.25 * R:
            add((hd + 0.01, 0, mxx, myy, 0.44 * R * S, pal['muzzle'], 0.0))
            if p['mouth'] > 0.05:
                X, Y, _ = HP(0.72 * R, -0.58 * R)
                add((hd + 0.011, 0, X, Y, 0.3 * R * S * p['mouth'], pal['dark'], 0.0))
                X, Y, _ = HP(0.72 * R, -0.66 * R)
                add((hd + 0.012, 0, X, Y, 0.16 * R * S * p['mouth'], pal['pink'], 0.0))
            X, Y, nd = HP(0.98 * R, -0.08 * R)
            if nd > -0.1 * R:
                add((hd + 0.02, 0, X, Y, 0.17 * R * S, pal['nose'], 0.0))
            if S >= 1.8:
                for k in (-1, 1):
                    A, Bp = HP(0.8 * R, -0.38 * R, 0.25 * R * k), HP(1.9 * R, (-0.25 + 0.2 * k) * R, 0.6 * R * k)
                    add((hd + 0.03, 1, A[0], A[1], Bp[0], Bp[1], 0.35, 0.3, pal['whisk'], 0.75))
        eo = p['eye'] * self.blink
        for l in (-0.4, 0.4):
            ex, ey, ed = HP(0.6 * R, 0.12 * R, l * R)
            if ed > 0.12 * R:
                er = 0.25 * R * S
                if eo < 0.22:
                    add((hd + 0.03, 1, ex - er, ey, ex + er * 0.8, ey + 0.2 * er, 0.3 * S, 0.3 * S, pal['dark'], 1.0))
                else:
                    h = er * min(1.0, eo)
                    iris = pal['eye' if l > 0 else 'eye2']
                    add((hd + 0.03, 1, ex - er * 0.55, ey, ex + er * 0.55, ey, h, h, iris, 1.0))
                    pw = er * (0.22 + 0.5 * p['pup'])
                    add((hd + 0.031, 1, ex, ey - h * 0.7, ex, ey + h * 0.7, pw, pw, pal['pupil'], 1.0))

        prims.sort(key=lambda q: q[0])
        for q in prims:
            if q[1] == 0:
                cv.circle(q[2], q[3], q[4], q[5], q[6])
            else:
                cv.capsule(q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9])

    # ----------------------------------------------------------- action primitives (generators)
    def wait(self, d):
        end = self.t + d
        while self.t < end:
            yield

    def until(self, cond, timeout):
        end = self.t + timeout
        while self.t < end and not cond():
            yield

    def tween(self, key, b, dur):
        a, t0 = self.tgt[key], self.t
        while self.t < t0 + dur:
            self.tgt[key] = lerp(a, b, smooth((self.t - t0) / dur))
            yield
        self.tgt[key] = b

    def stand_up(self, quick=False):
        if self.posture == 'stand':
            return
        lying = self.posture in ('curl', 'loaf')
        self.set_pose('stand')
        yield from self.wait(0.22 if quick else (0.7 if lying else 0.45))

    def aim(self, d):
        """Pick the heading goal for facing d, turning toward the viewer (usually) or away."""
        self.dir = d
        base = 0.0 if d > 0 else PI
        k = math.floor((self.th - base) / TAU)
        lo, hi = base + k * TAU, base + (k + 1) * TAU
        toward = self.rng.random() < 0.7
        self.th_goal = hi if (math.sin((self.th + hi) / 2) > 0) == toward else lo

    def face(self, d):
        if d == self.dir:
            return
        self.vt = 0.0
        yield from self.until(lambda: self.v < 0.3, 1.5)
        # the front walks round while the hips nearly pivot in place, so the
        # body ends up a few steps along instead of spinning on its centre
        pivot = self.x - 4 * self.S * math.cos(self.th)
        lo, hi = self.sim.world.bounds(self.level)
        self.aim(d)
        g, t0 = self.th_goal, self.t
        a0h, a0f, a0r = self.thh, self.th, self.thr
        dur = self.rng.uniform(1.05, 1.35) * TURN_K / self.th_k
        self.turning = True
        while True:
            u = min(1.0, (self.t - t0) / dur)
            # head first, shoulders swing well past halfway before the hips follow
            self.thh = lerp(a0h, g, smooth(u / 0.4))
            self.th = lerp(a0f, g, smooth(u / 0.68))
            self.thr = lerp(a0r, g, smooth((u - 0.3) / 0.7))
            self.dip = math.sin(PI * u)
            self.x = clamp(pivot + 4 * self.S * math.cos((self.th + self.thr) / 2), lo, hi)
            if u >= 1:
                break
            yield
        self.turning, self.dip = False, 0.0
        self.thhv = self.thv = self.thrv = 0.0
        k = round(self.th_goal / TAU)
        if k:
            self.th -= k * TAU
            self.thr -= k * TAU
            self.thh -= k * TAU
            self.th_goal -= k * TAU

    def walk_to(self, tx, speed=None, gal=0.0, pauses=True, pose='stand'):
        S, rng = self.S, self.rng
        tx = clamp(tx, *self.sim.world.bounds(self.level))
        if abs(tx - self.x) < 1.0 * S:
            return
        if self.posture != pose:
            yield from self.stand_up(quick=True)
        d = 1 if tx > self.x else -1
        yield from self.face(d)
        speed = speed or rng.uniform(7.5, 11)
        self.set_pose(pose)
        self.tgt['gal'], self.tgt['stride'] = gal, 1.0
        self.acc, self.dec = 10 + 45 * gal, 14 + 50 * gal
        next_pause = self.t + rng.uniform(2.5, 7)
        while True:
            dist = (tx - self.x) * d / S
            if dist <= 0.15:
                break
            wob = 1 + 0.18 * noise(self.t * 0.8 + self.seed)
            self.vt = max(0.8, min(speed * wob, math.sqrt(2 * self.dec * dist)))
            if pauses and self.t > next_pause and dist > 12:
                # stop mid-walk (often with a paw still up), glance at something, carry on
                self.vt = 0.0
                yield from self.until(lambda: self.v < 0.2, 2)
                hx, hy = self.head_pos()
                self.look = rng.choice(['viewer', (hx + rng.uniform(-30, 30) * S, hy + rng.uniform(-15, 12) * S),
                                        (self.sim.world.clock_x, self.sim.world.clock_y)])
                yield from self.wait(rng.uniform(0.6, 2.2))
                self.look = None
                next_pause = self.t + rng.uniform(3, 9)
            yield
        self.vt = 0.0
        self.tgt['stride'] = 0.0
        yield from self.until(lambda: self.v < 0.1, 1.0)
        self.v = 0.0

    def go(self, tx, level=None, **kw):
        w, S = self.sim.world, self.S
        level = level or self.level
        if level != self.level:
            if self.level == 'floor':
                lx = clamp(tx, *w.bounds('sill'))
                side = -1 if self.x < lx else 1
                jx = clamp(lx + side * 11 * S, *w.bounds('floor'))
                yield from self.walk_to(jx, **kw)
                yield from self.jump_to(lx, 'sill', h=4)
            else:
                b0, b1 = w.bounds('sill')
                edge = b1 if tx > (b0 + b1) / 2 else b0
                yield from self.walk_to(edge, **kw)
                land = clamp(edge + (1 if edge == b1 else -1) * 13 * S, *w.bounds('floor'))
                yield from self.jump_to(land, 'floor', h=3)
        yield from self.walk_to(tx, **kw)

    def jump_to(self, tx, level, h=5.0, anticip=(0.25, 0.55), air='air', land='crouch', flip=False):
        S, rng = self.S, self.rng
        w = self.sim.world
        d = 1 if tx > self.x + 0.5 * S else (-1 if tx < self.x - 0.5 * S else self.dir)
        yield from self.stand_up(quick=True)
        yield from self.face(d)
        self.busy = True
        y1 = w.ground(level, tx)
        self.look = (tx + d * 6 * S, y1 - 4 * S)
        self.set_pose('crouch', tailA=0.2)
        yield from self.wait(rng.uniform(*anticip))
        self.look = None
        x0, y0 = self.x, self.y
        top = min(y0, y1) - h * S
        hh = (y0 + y1) / 2 - top
        dur = 0.32 + 0.011 * abs(tx - x0) / S + 0.014 * max(0.0, (y0 - y1) / S) + 0.01 * h
        self.set_pose(air)
        base_sy, base_hy = self.tgt['sy'], self.tgt['hy']
        self.airborne, self.v, self.vt, self.tgt['stride'] = True, 0.0, 0.0, 0.0
        self.snap = 1.6
        t0, flipped = self.t, False
        while True:
            u = (self.t - t0) / dur
            if u >= 1:
                break
            self.x = lerp(x0, tx, u)
            self.y = lerp(y0, y1, u) - 4 * hh * u * (1 - u)
            vyu = (y1 - y0) - 4 * hh * (1 - 2 * u)
            pitch = clamp(math.atan2(-vyu, abs(tx - x0) + 6 * S), -0.7, 0.7) if air == 'air' else 0.0
            self.tgt['sy'], self.tgt['hy'] = base_sy + pitch * 4, base_hy - pitch * 4
            if u > 0.55 and air == 'air':  # reach for the landing
                self.tgt.update(ffx=8.0, ffy=1.5, hfx=-7.0, hfy=3.0)
            if flip and not flipped and u > 0.35:
                self.aim(-self.dir)
                self.th_k, flipped = 16.0, True
            yield
        self.x, self.y, self.level, self.airborne = tx, y1, level, False
        self.sim.dust(self.x, y1, 6)
        self.set_pose(land)
        self.snap = 1.3
        yield from self.wait(0.16)
        self.busy, self.snap, self.th_k = False, 1.0, TURN_K
        if land == 'crouch':
            self.set_pose('stand')
            yield from self.wait(0.2)

    def slow_blink(self):
        self.look = 'viewer'
        yield from self.wait(self.rng.uniform(0.5, 0.9))
        yield from self.tween('eye', 0.18, 0.9)
        yield from self.wait(self.rng.uniform(0.4, 0.9))
        yield from self.tween('eye', 1.0, 0.6)
        if self.rng.random() < 0.45:
            hx, hy = self.head_pos()
            self.sim.say(hx, hy, '<3', (255, 120, 140))
        yield from self.wait(0.6)
        self.look = None

    def look_around(self, n=None):
        hx, hy = self.head_pos()
        for _ in range(n or self.rng.randint(2, 3)):
            self.look = (hx + self.rng.uniform(-40, 40) * self.S, hy + self.rng.uniform(-20, 10) * self.S)
            yield from self.wait(self.rng.uniform(0.35, 0.8))
        self.look = None

    # ----------------------------------------------------------- behaviours
    def b_sit(self):
        rng, w = self.rng, self.sim.world
        self.doing = 'sitting'
        if self.posture != 'sit':
            self.set_pose('sit')
            yield from self.wait(0.6)
        end = self.t + rng.uniform(7, 22)
        while self.t < end:
            r = rng.random()
            if r < 0.22:
                yield from self.slow_blink()
            elif r < 0.32:
                self.look = (w.clock_x, w.clock_y)
            elif r < 0.45 and w.bird.state != 'gone':
                self.look = lambda: (w.bird.x, w.bird.y)
            elif r < 0.75:
                hx, hy = self.head_pos()
                self.look = (hx + rng.uniform(-40, 40) * self.S, hy + rng.uniform(-25, 10) * self.S)
            else:
                self.look = None
            self.tail_freq = rng.uniform(0.15, 0.4)
            yield from self.wait(rng.uniform(1.2, 4.0))
        self.look = None

    def b_loaf(self):
        rng = self.rng
        self.doing = 'loafing'
        if self.posture != 'loaf':
            if self.posture != 'sit':
                self.set_pose('sit')
                yield from self.wait(0.5)
            self.set_pose('loaf')
            yield from self.wait(0.8)
        end = self.t + rng.uniform(12, 40)
        while self.t < end:
            if rng.random() < 0.3:
                yield from self.slow_blink()
                self.tgt['eye'] = 0.75
            yield from self.wait(rng.uniform(2, 5))
            if self.energy < 0.45 and rng.random() < 0.3:
                yield from self.sleep_here('loaf')
                yield from self.b_wake()
                return

    def b_groom(self):
        rng = self.rng
        self.doing = 'grooming'
        if self.posture != 'sit':
            yield from self.stand_up(quick=True)
            self.set_pose('sit')
            yield from self.wait(0.6)
        for _ in range(rng.randint(2, 4)):
            kind = rng.choice(('paw', 'paw', 'face', 'flank'))
            if kind == 'flank':
                self.tgt.update(hyaw=1.9, htilt=-0.55, hdy=4.0)
                yield from self.wait(0.5)
                t0, dur = self.t, rng.uniform(2, 3.5)
                while self.t < t0 + dur:
                    k = math.sin((self.t - t0) * 11)
                    self.tgt['mouth'] = 0.3 + 0.25 * k
                    self.tgt['htilt'] = -0.55 + 0.12 * k
                    yield
                self.tgt.update(hyaw=0.0, htilt=0.0, hdy=POSES['sit']['hdy'], mouth=0.0)
            else:
                self.pext[0], self.peyt[0] = 1.0, 11.0
                self.tgt.update(htilt=-0.45, hdx=2.4, hdy=4.8)
                yield from self.wait(0.4)
                t0, dur = self.t, rng.uniform(1.4, 2.8)
                while self.t < t0 + dur:
                    k = math.sin((self.t - t0) * 12)
                    self.tgt['mouth'] = 0.3 + 0.3 * k
                    self.tgt['hdy'] = 4.8 + 0.25 * k
                    yield
                self.tgt['mouth'] = 0.0
                if kind == 'face':
                    for _ in range(rng.randint(2, 3)):
                        self.tgt.update(htilt=-0.75, ear=0.6)
                        self.pext[0], self.peyt[0] = 0.4, 14.5
                        yield from self.wait(0.35)
                        self.pext[0], self.peyt[0] = 1.4, 11.0
                        yield from self.wait(0.3)
                    self.tgt['ear'] = 0.0
                self.pext[0] = self.peyt[0] = 0.0
                self.tgt.update(htilt=0.0, hdx=POSES['sit']['hdx'], hdy=POSES['sit']['hdy'])
            yield from self.wait(rng.uniform(0.4, 1.4))
        if rng.random() < 0.4:
            yield from self.slow_blink()

    def b_wander(self):
        rng, w, S = self.rng, self.sim.world, self.S
        self.doing = 'wandering'
        lvl = self.level
        if rng.random() < 0.25:
            lvl = 'sill' if lvl == 'floor' else 'floor'
        lo, hi = w.bounds(lvl)
        tx = rng.uniform(lo, hi)
        if lvl == self.level and abs(tx - self.x) < 15 * S:
            tx = hi if self.x < (lo + hi) / 2 else lo
        yield from self.go(tx, lvl)
        if rng.random() < 0.5:
            self.doing = 'sniffing'
            self.tgt.update(hdy=1.2, htilt=-0.7, hdx=4.6)
            yield from self.wait(rng.uniform(1, 2.2))
            self.tgt.update(hdy=POSE_BASE['hdy'], htilt=0.0, hdx=POSE_BASE['hdx'])
            yield from self.wait(0.4)

    def b_nap(self):
        rng, w, S = self.rng, self.sim.world, self.S
        self.doing = 'looking for a spot'
        r = rng.random()
        if self.level == 'sill' and r < 0.5:
            spot, lvl, pose = clamp(self.x, *w.bounds('sill')), 'sill', 'loaf'
        else:
            spot, lvl, pose = w.bed_x + rng.uniform(-2, 2) * S, 'floor', 'curl'
        yield from self.go(spot, lvl)
        if abs(self.x - w.bed_x) < w.bed_hw:
            # knead the blanket, then turn round once before settling
            self.doing = 'kneading'
            self.set_pose('stand', eye=0.45, tailA=1.3)
            yield from self.wait(0.4)
            t0, dur = self.t, rng.uniform(3, 6)
            while self.t < t0 + dur:
                k = math.sin((self.t - t0) * 7)
                self.peyt[0], self.peyt[1] = max(0, k) * 1.6, max(0, -k) * 1.6
                self.tgt['hy'] = 8.4 + 0.25 * abs(k)
                if rng.random() < self.dt * 0.4:
                    hx, hy = self.head_pos()
                    self.sim.say(hx, hy, '~', (255, 200, 150))
                yield
            self.peyt[0] = self.peyt[1] = 0.0
            yield from self.face(-self.dir)
            yield from self.wait(0.2)
            yield from self.face(-self.dir)
        self.set_pose('sit' if pose == 'loaf' else 'loaf')
        yield from self.wait(0.6)
        yield from self.sleep_here(pose)
        yield from self.b_wake()

    def sleep_here(self, pose):
        rng = self.rng
        self.set_pose(pose, eye=0.0)
        self.asleep = True
        self.doing = 'napping'
        end = self.t + lerp(220, 45, self.energy) * rng.uniform(0.75, 1.25)
        next_z, next_evt = self.t + 2.5, self.t + rng.uniform(5, 14)
        while self.t < end and self.energy < 0.995:
            if self.t > next_z:
                hx, hy = self.head_pos()
                self.sim.say(hx + 1.5 * self.S, hy, 'z', (200, 210, 255))
                next_z = self.t + rng.uniform(2.5, 4.5)
            if self.t > next_evt:
                ev = rng.random()
                if ev < 0.35:  # dream: little paw twitches
                    for _ in range(rng.randint(2, 4)):
                        self.peyt[0], self.pext[0] = 0.8, 0.6
                        yield from self.wait(0.12)
                        self.peyt[0] = self.pext[0] = 0.0
                        yield from self.wait(rng.uniform(0.1, 0.3))
                elif ev < 0.6:
                    self.tail_tw = 1.2
                    yield from self.wait(0.6)
                    self.tail_tw = 0.0
                elif ev < 0.8:
                    self.tgt['rb'] = POSES[pose].get('rb', 1) * 1.06  # sigh
                    yield from self.wait(1.2)
                    self.tgt['rb'] = POSES[pose].get('rb', 1)
                else:  # half-wake, check the room, resettle
                    self.tgt.update(hdy=3.2, htilt=0.15, eye=0.35)
                    yield from self.wait(0.8)
                    yield from self.look_around(2)
                    self.set_pose(pose, eye=0.0)
                next_evt = self.t + rng.uniform(5, 16)
            yield
        self.asleep = False

    def b_wake(self):
        rng = self.rng
        self.doing = 'waking up'
        self.asleep = False
        yield from self.tween('eye', 0.45, 0.8)
        yield from self.wait(rng.uniform(0.4, 1.0))
        if self.posture == 'curl':
            self.set_pose('loaf', eye=0.6)
            yield from self.wait(0.8)
        yield from self.look_around(2)
        self.tgt.update(mouth=1.0, eye=0.0, ear=0.6, htilt=0.55)  # yawn
        yield from self.wait(rng.uniform(1.1, 1.7))
        self.tgt.update(mouth=0.0, eye=1.0, ear=0.0, htilt=0.0)
        yield from self.wait(0.4)
        self.set_pose('stand')
        yield from self.wait(0.6)
        self.doing = 'stretching'
        self.set_pose('stretch')
        yield from self.wait(rng.uniform(1.4, 2.4))
        self.set_pose('stand')
        yield from self.wait(0.5)
        self.pext[2], self.peyt[2] = -4.5, 1.8  # hind-leg stretch
        self.tgt['sx'] += 1.0
        yield from self.wait(0.9)
        self.pext[2] = self.peyt[2] = 0.0
        self.set_pose('stand')
        yield from self.wait(0.4)
        if rng.random() < 0.5:
            yield from self.b_groom()

    def b_window(self):
        rng, w, S = self.rng, self.sim.world, self.S
        self.doing = 'gazing outside'
        lo, hi = w.bounds('sill')
        tx = clamp(lerp(w.wx0, w.wx1, rng.uniform(0.3, 0.6)), lo, hi)
        yield from self.go(tx, 'sill')
        self.set_pose('sit')
        yield from self.wait(0.6)
        bird = w.bird
        end = self.t + rng.uniform(15, 45)
        self.look = ((w.wx0 + w.wx1) / 2, (w.wy0 + w.wy1) / 2, True)
        while self.t < end:
            if bird.state != 'gone':
                self.look = lambda: (bird.x, bird.y, True)
                self.tgt['pup'] = 1.0
                if bird.state == 'perch':
                    self.doing = 'chattering at a bird'
                    self.tgt['mouth'] = (0.25 + 0.2 * math.sin(self.t * 32)) if (self.t % 2.4) < 1.1 else 0.0
                    self.tail_tw, self.tail_freq = 0.9, 0.6
                    end = max(end, self.t + 2)
                    if abs(bird.x - self.x) < 16 * S and rng.random() < self.dt * 0.12:
                        self.pext[0], self.peyt[0] = 1.5, 8.5   # paw at the glass
                        yield from self.wait(0.35)
                        bird.spook()
                        self.pext[0] = self.peyt[0] = 0.0
                else:
                    self.tgt['mouth'], self.tail_tw = 0.0, 0.3
            else:
                self.doing = 'gazing outside'
                self.tgt['mouth'], self.tail_tw, self.tgt['pup'] = 0.0, 0.0, 0.3
                if rng.random() < self.dt * 0.25:
                    self.look = (rng.uniform(w.wx0, w.wx1), rng.uniform(w.wy0, w.wy1), True)
                elif rng.random() < self.dt * 0.05:
                    self.look = 'viewer'
            yield
        self.tgt['mouth'], self.tail_tw, self.tgt['pup'] = 0.0, 0.0, 0.3
        self.look = None
        if (w.day > 0.6 or self.energy < 0.5) and rng.random() < 0.6:
            self.doing = 'sunbathing' if w.day > 0.6 else 'loafing'
            self.set_pose('loaf')
            yield from self.wait(rng.uniform(8, 20))
            if self.energy < 0.6 and rng.random() < 0.5:
                yield from self.sleep_here('loaf')
                yield from self.b_wake()

    def b_knock(self):
        rng, sim, S = self.rng, self.sim, self.S
        mug = sim.mug
        if mug.state != 'sill':
            return
        self.doing = 'plotting'
        lo, hi = sim.world.bounds('sill')
        yield from self.go(clamp(mug.x - 8.8 * S, lo, hi), 'sill')
        yield from self.face(1)
        if mug.state != 'sill':
            return
        self.set_pose('sit')
        yield from self.wait(0.6)
        self.look = (mug.x, mug.y - mug.h)
        yield from self.wait(rng.uniform(0.8, 1.5))
        self.look = 'viewer'   # making sure you're watching
        yield from self.wait(rng.uniform(1.0, 2.0))
        self.doing = 'testing gravity'
        taps = rng.randint(2, 5)
        for i in range(taps + 4):
            if mug.state != 'sill':
                break
            self.look = lambda: (mug.x, mug.y - mug.h)
            reach = (mug.x - mug.hw - self.x) / S - POSES['sit']['ffx'] - 0.9
            self.pext[0], self.peyt[0] = reach + 0.6, 1.8
            yield from self.wait(0.22)
            mug.push((2.6 if i >= taps else rng.uniform(0.35, 0.6)) * S)
            self.pext[0], self.peyt[0] = reach - 1.2, 1.0
            yield from self.wait(0.25)
            self.pext[0] = self.peyt[0] = 0.0
            if rng.random() < 0.55:
                self.look = 'viewer'
                yield from self.wait(rng.uniform(0.7, 1.6))
            else:
                yield from self.wait(rng.uniform(0.3, 0.8))
        self.doing = 'watching it fall'
        self.look = lambda: (mug.x, mug.y)
        yield from self.until(lambda: mug.state != 'fall', 3)
        yield from self.wait(0.8)
        self.mischief = 0.0
        self.doing = 'pleased'
        yield from self.slow_blink()

    def b_yarn(self):
        rng, sim, S = self.rng, self.sim, self.S
        Y = sim.yarn
        self.doing = 'playing'
        lo, hi = sim.world.bounds('floor')
        if self.level != 'floor':
            yield from self.go(clamp(Y.x, lo, hi), 'floor')
        bats, fails = rng.randint(3, 6), 0
        end = self.t + rng.uniform(35, 70)
        self.tgt['pup'] = 0.8
        while bats > 0 and self.t < end and fails < 4:
            self.look = lambda: (Y.x, Y.y)
            if abs(Y.vx) > 6 * S or Y.air:
                yield from self.wait(0.15)
                continue
            d = 1 if Y.x > self.x else -1
            tx = clamp(Y.x - d * 9.2 * S, lo, hi)
            far = abs(tx - self.x) / S
            if far > 26 and rng.random() < 0.5:
                self.doing = 'stalking the yarn'
                yield from self.walk_to(Y.x - d * 20 * S, speed=6, pose='crouch', pauses=False)
                if abs(Y.vx) > 2 * S:
                    continue
                yield from self.wiggle(rng.uniform(0.4, 0.9))
                yield from self.jump_to(clamp(Y.x - self.dir * 8.6 * S, lo, hi), 'floor', h=3.0,
                                        anticip=(0.05, 0.1), land='pin')
                if abs(Y.x - (self.x + self.dir * 8.6 * S)) < 3.5 * S:
                    Y.kick(self.dir * rng.uniform(15, 35) * S, up=rng.uniform(10, 25) * S)
                bats -= 1
                self.doing = 'playing'
                yield from self.wait(0.4)
                continue
            if far > 1.5:
                yield from self.walk_to(tx, speed=18 if far > 20 else 8, gal=0.7 if far > 30 else 0.0, pauses=False)
                fails += abs(self.x - tx) > 2 * S
                continue
            if d != self.dir:
                yield from self.face(d)
                continue
            reach = (Y.x - self.x) * self.dir / S
            if not 6.5 < reach < 12:
                fails += 1
                yield from self.wait(0.3)
                continue
            if rng.random() < 0.3:
                self.tgt.update(hdy=2.0, htilt=-0.6)
                yield from self.wait(0.6)
                self.tgt.update(hdy=POSE_BASE['hdy'], htilt=0.0)
            self.pext[0], self.peyt[0] = reach - POSE_BASE['ffx'] - 0.8, 2.6
            yield from self.wait(0.13)
            Y.kick(self.dir * rng.uniform(22, 55) * S, up=(rng.uniform(8, 20) * S if rng.random() < 0.3 else 0))
            self.pext[0], self.peyt[0] = 0.5, 0.4
            yield from self.wait(0.2)
            self.pext[0] = self.peyt[0] = 0.0
            bats -= 1
            yield from self.until(lambda: abs(Y.vx) < 3 * S and not Y.air, 2.5)
        self.look = None
        self.tgt['pup'] = 0.3
        self.boredom = max(0.0, self.boredom - 0.5)
        self.set_pose('sit')
        yield from self.wait(0.6)
        if rng.random() < 0.5:
            yield from self.b_groom()

    def wiggle(self, dur):
        self.set_pose('crouch', hy=5.0)
        t0 = self.t
        self.tail_tw = 1.0
        while self.t < t0 + dur:
            k = math.sin((self.t - t0) * 16)
            self.tgt['hy'] = 5.2 + 0.5 * k
            self.peyt[2], self.peyt[3] = 0.5 * max(0, k), 0.5 * max(0, -k)
            yield
        self.peyt[2] = self.peyt[3] = 0.0
        self.tail_tw = 0.4

    def b_zoomies(self):
        rng, w, S = self.rng, self.sim.world, self.S
        self.doing = 'ZOOMIES'
        if self.level != 'floor':
            yield from self.go(lerp(*w.bounds('floor'), 0.5), 'floor')
        yield from self.stand_up(quick=True)
        self.set_pose('crouch', ear=0.8, pup=1.0)
        yield from self.wait(0.3)
        yield from self.wiggle(0.4)
        lo, hi = w.bounds('floor')
        for _ in range(rng.randint(2, 4)):
            d = 1 if hi - self.x > self.x - lo else -1
            self.th_k = 16.0
            yield from self.face(d)
            wall = hi if d > 0 else lo
            reaction = rng.choice(('skid', 'skid', 'kick'))
            stop = 18 * S if reaction == 'kick' else 0.0
            self.set_pose('stand', ear=0.8, pup=1.0, tailA=0.6)
            self.tgt['gal'], self.tgt['stride'] = 1.0, 1.0
            self.acc, self.dec = 55.0, 75.0
            speed = rng.uniform(34, 42)
            while (wall - self.x) * d > stop + (self.v ** 2 / (2 * self.dec)) * S * (reaction == 'skid'):
                self.vt = speed * (1 + 0.1 * noise(self.t * 2))
                yield
            if reaction == 'kick':
                yield from self.jump_to(wall + d * 2 * S, 'floor', h=11, anticip=(0.0, 0.02), flip=True)
            else:
                self.doing = 'ZOOMIES (skidding)'
                self.set_pose('skid')
                self.vt = 0.0
                while self.v > 0.6:
                    if rng.random() < 0.6:
                        self.sim.dust(self.x + d * 9 * S, self.y, 1)
                    yield
                self.tgt['stride'] = 0.0
                self.doing = 'ZOOMIES'
                yield from self.face(-d)
            self.th_k = TURN_K
            if rng.random() < 0.4:  # sudden freeze mid-room
                self.set_pose('stand', ear=0.0, pup=1.0)
                self.tgt['stride'] = 0.0
                yield from self.look_around(2)
        self.vt = 0.0
        yield from self.until(lambda: self.v < 0.2, 1.5)
        self.tgt['stride'], self.tgt['gal'] = 0.0, 0.0
        self.zoom_cool = self.t + rng.uniform(200, 500)
        self.energy = max(0.0, self.energy - 0.15)
        self.set_pose('sit')
        self.doing = 'acting normal'
        yield from self.wait(0.6)
        yield from self.b_groom()

    def b_hunt(self):
        rng, sim, S = self.rng, self.sim, self.S
        L, w = sim.laser, sim.world
        self.doing = 'hunting'
        if self.asleep:
            self.asleep = False
            self.set_pose('loaf', eye=1.0)
            yield from self.wait(0.35)
        self.look = lambda: (L.x, L.y)
        self.tgt['pup'] = 1.0
        self.tail_tw = 0.5
        yield from self.wait(rng.uniform(0.3, 0.8))
        interest = 0.55 + 0.35 * self.energy + 0.3 * self.boredom
        end, last = self.t + rng.uniform(45, 90), self.t
        lo, hi = w.bounds('floor')
        while L.on and self.t < end and interest > 0.15:
            interest -= (self.t - last) / 80
            last = self.t
            lx = L.x
            on_wall = L.y < w.floor_top - 0.5 * S
            if self.level != 'floor':
                yield from self.go(clamp(lx, lo, hi), 'floor', pauses=False)
                continue
            dx = lx - self.x
            d = 1 if dx > 0 else -1
            dist = abs(dx) / S
            if d != self.dir and dist > 3:
                yield from self.face(d)
                continue
            if on_wall:
                if dist > 5 and lo < lx - d * 3 * S < hi:
                    yield from self.walk_to(lx - d * 3 * S, speed=12, pauses=False)
                    continue
                self.set_pose('sit')
                yield from self.wait(rng.uniform(0.6, 1.4))
                height = (self.y - L.y) / S
                if L.on and height < 26 and rng.random() < 0.55:
                    yield from self.jump_to(self.x, 'floor', h=clamp(height - 10, 3, 14), anticip=(0.3, 0.6),
                                            air='reach')
                    interest -= 0.1
                else:
                    interest -= 0.05
                continue
            if dist > 34:
                yield from self.walk_to(lx - d * 22 * S, speed=26, gal=0.9, pauses=False)
                continue
            if dist > 14:
                self.doing = 'stalking'
                if self.posture != 'crouch':
                    self.set_pose('crouch')
                self.tgt['stride'], self.tgt['gal'] = 1.0, 0.0
                self.acc, self.dec = 12.0, 30.0
                self.vt = rng.uniform(3.5, 6.0)
                yield from self.until(lambda: abs(L.x - lx) > 8 * S or abs(L.x - self.x) / S < 13
                                      or (L.x - self.x) * self.dir < 0, rng.uniform(0.5, 1.4))
                self.vt = 0.0
                yield from self.until(lambda: self.v < 0.2, 0.8)
                yield from self.wait(rng.uniform(0.2, 0.9))   # frozen, one paw up
                continue
            self.vt = 0.0
            self.tgt['stride'] = 0.0
            yield from self.until(lambda: self.v < 0.2, 0.8)
            self.doing = 'about to pounce'
            moved = False
            for _ in self.wiggle(rng.uniform(0.5, 1.3)):
                if abs(L.x - lx) > 6 * S:
                    moved = True
                    break
                yield
            self.tail_tw = 0.4
            if moved or not L.on:
                continue
            tx = clamp(L.x - d * 8.6 * S, lo, hi)
            L.pounced(self.t)
            yield from self.jump_to(tx, 'floor', h=3.5, anticip=(0.04, 0.08), land='pin')
            interest -= rng.uniform(0.08, 0.18)
            paw = self.x + self.dir * 8.6 * S
            if L.on and abs(L.x - paw) < 3 * S:
                self.doing = 'got it?'
                self.look = (paw, self.y)
                yield from self.wait(rng.uniform(0.7, 1.3))
                self.peyt[0] = 2.0
                yield from self.wait(0.6)
                L.vanish(self.t)
                hx, hy = self.head_pos()
                sim.say(hx, hy, '?', (255, 240, 160))
                yield from self.wait(0.6)
                self.peyt[0] = 0.0
                self.look = lambda: (L.x, L.y)
            self.doing = 'hunting'
            self.set_pose('crouch')
            yield from self.wait(rng.uniform(0.2, 0.6))
        self.tail_tw = 0.0
        self.tgt['pup'] = 0.3
        self.set_pose('sit')
        if not L.on:
            hx, hy = self.head_pos()
            sim.say(hx, hy, '?', (255, 240, 160))
            self.doing = 'confused'
            yield from self.look_around(3)
        else:
            self.doing = 'over it'
            self.hunt_cool = self.t + rng.uniform(25, 50)
            yield from self.wait(rng.uniform(2, 5))
            self.look = None
        self.boredom = max(0.0, self.boredom - 0.3)
        yield from self.wait(0.8)

    def b_startle(self):
        rng = self.rng
        self.doing = 'startled'
        hx, hy = self.head_pos()
        self.sim.say(hx, hy - 2 * self.S, '!', (255, 230, 120))
        if self.asleep:
            self.asleep = False
            self.snap = 2.0
            self.set_pose('loaf', eye=1.0, ear=0.0, pup=1.0, hdy=5.2)
            yield from self.wait(0.6)
            self.snap = 1.0
            yield from self.look_around(3)
            return
        self.busy = True
        self.snap = 2.2
        x0 = self.x
        self.set_pose('arch')
        yield from self.wait(0.12)  # fur up first, then the hop
        self.airborne = True
        t0 = self.t
        while self.t < t0 + 0.42:
            u = (self.t - t0) / 0.42
            self.y = self.sim.world.ground(self.level, x0) - 6 * self.S * 4 * u * (1 - u)
            yield
        self.airborne, self.busy, self.snap = False, False, 1.0
        self.sim.dust(self.x, self.y, 4)
        yield from self.wait(rng.uniform(0.6, 1.2))
        yield from self.look_around(2)
        self.set_pose('stand')
        yield from self.wait(0.6)
        if rng.random() < 0.6:
            yield from self.b_groom()


# ------------------------------------------------------------------ fastfetch
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ''


def _gib(kib):
    return '%.2f GiB' % (kib / 1048576)


class SysInfo:
    """fastfetch-style facts read straight from /proc and /sys (Linux), degrading quietly elsewhere."""
    ORDER = ('OS', 'Host', 'Kernel', 'Uptime', 'Packages', 'Shell', 'Terminal', 'CPU', 'GPU', 'Memory', 'Swap',
             'Disk (/)', 'Load', 'Battery', 'Locale')

    def __init__(self):
        import platform
        import socket
        import getpass
        import subprocess
        import re
        self.user = os.environ.get('USER') or getpass.getuser()
        self.host = socket.gethostname().split('.')[0]
        f = self.facts = {}
        osr = dict(l.split('=', 1) for l in _read('/etc/os-release').splitlines() if '=' in l)
        name = osr.get('PRETTY_NAME', '').strip('"') or (('macOS ' + platform.mac_ver()[0]) if sys.platform == 'darwin'
                                                         else platform.system())
        f['OS'] = '%s %s' % (name, platform.machine())
        prod = _read('/sys/devices/virtual/dmi/id/product_name')
        if not prod or 'filled' in prod or prod in ('System Product Name', 'Default string'):
            prod = ' '.join(filter(None, (_read('/sys/devices/virtual/dmi/id/board_vendor'),
                                          _read('/sys/devices/virtual/dmi/id/board_name'))))
        f['Host'] = prod
        f['Kernel'] = '%s %s' % (platform.system(), platform.release())
        pk = []
        try:
            n = sum(1 for e in os.listdir('/var/lib/dpkg/info') if e.endswith('.list'))
            pk.append('%d (dpkg)' % n)
        except OSError:
            pass
        for path, label, skip in (('/var/lib/pacman/local', 'pacman', ('ALPM_DB_VERSION',)),
                                  ('/var/lib/flatpak/app', 'flatpak', ()), ('/snap', 'snap', ('bin', 'README'))):
            try:
                n = sum(1 for e in os.listdir(path) if e not in skip)
                if n:
                    pk.append('%d (%s)' % (n, label))
            except OSError:
                pass
        f['Packages'] = ', '.join(pk)
        sh = os.environ.get('SHELL', '')
        ver = ''
        if sh:
            try:
                out = subprocess.run([sh, '--version'], capture_output=True, text=True, timeout=0.5).stdout
                m = re.search(r'(\d+\.\d+(\.\d+)?)', out)
                ver = m.group(1) if m else ''
            except (OSError, subprocess.SubprocessError):
                pass
        f['Shell'] = ('%s %s' % (os.path.basename(sh), ver)).strip()
        term = os.environ.get('TERM_PROGRAM', '')
        if term:
            term += ' ' + os.environ.get('TERM_PROGRAM_VERSION', '')
        else:  # walk up the process tree past shells to the terminal
            pid = os.getppid()
            for _ in range(8):
                comm = _read('/proc/%d/comm' % pid)
                if comm and comm not in ('zsh', 'bash', 'sh', 'fish', 'dash', 'python', 'python3', 'sudo', 'su',
                                         'login', 'timeout', 'env'):
                    term = comm
                    break
                stat = _read('/proc/%d/stat' % pid).rsplit(')', 1)
                if len(stat) < 2:
                    break
                pid = int(stat[1].split()[1])
            term = term or os.environ.get('TERM', '')
        f['Terminal'] = term.strip()
        cpu = next((l.split(':', 1)[1] for l in _read('/proc/cpuinfo').splitlines() if l.startswith('model name')),
                   platform.processor())
        cpu = re.sub(r'\((R|TM)\)|CPU|@.*', '', cpu)
        cpu = ' '.join(cpu.split()) + ' (%d)' % (os.cpu_count() or 1)
        mhz = _read('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq')
        f['CPU'] = cpu + (' @ %.2f GHz' % (int(mhz) / 1e6) if mhz.isdigit() else '')
        gpu = ''
        try:
            out = subprocess.run(['lspci'], capture_output=True, text=True, timeout=1).stdout
            for l in out.splitlines():
                if any(k in l for k in ('VGA', '3D controller', 'Display controller')):
                    desc = l.split(': ', 1)[1]
                    m = re.search(r'\[([^\]]+)\]', desc)
                    gpu = desc.split()[0] + ' ' + m.group(1) if m else desc
                    break
        except (OSError, subprocess.SubprocessError):
            pass
        f['GPU'] = gpu
        f['Locale'] = os.environ.get('LANG', '')
        self.last = -99.0
        self.refresh(0.0)

    def refresh(self, t):
        if t - self.last < 2.0:
            return
        self.last = t
        f = self.facts
        up = _read('/proc/uptime').split()
        if up:
            m = int(float(up[0]) // 60)
            parts = [(m // 1440, 'day'), (m // 60 % 24, 'hour'), (m % 60, 'min')]
            f['Uptime'] = ', '.join('%d %s%s' % (v, u, 's' if v != 1 else '') for v, u in parts if v) or '0 mins'
        mem = {}
        for l in _read('/proc/meminfo').splitlines():
            k, _, v = l.partition(':')
            mem[k] = int(v.split()[0]) if v.split() else 0
        self.mem_pct = 0
        if mem.get('MemTotal'):
            used = mem['MemTotal'] - mem.get('MemAvailable', 0)
            self.mem_pct = 100 * used // mem['MemTotal']
            f['Memory'] = '%s / %s (%d%%)' % (_gib(used), _gib(mem['MemTotal']), self.mem_pct)
        if mem.get('SwapTotal'):
            used = mem['SwapTotal'] - mem.get('SwapFree', 0)
            f['Swap'] = '%s / %s (%d%%)' % (_gib(used), _gib(mem['SwapTotal']), 100 * used // mem['SwapTotal'])
        try:
            import shutil
            d = shutil.disk_usage('/')
            f['Disk (/)'] = '%s / %s (%d%%)' % (_gib(d.used / 1024), _gib(d.total / 1024), 100 * d.used // d.total)
        except OSError:
            pass
        try:
            self.load = os.getloadavg()
            f['Load'] = '%.2f, %.2f, %.2f' % self.load
        except OSError:
            self.load = (0.0, 0.0, 0.0)
        try:
            bats = [b for b in os.listdir('/sys/class/power_supply') if b.startswith('BAT')]
        except OSError:
            bats = []
        if bats:
            p = '/sys/class/power_supply/' + bats[0]
            f['Battery'] = '%s%% [%s]' % (_read(p + '/capacity'), _read(p + '/status'))

    def lines(self):
        return [(k, self.facts[k]) for k in self.ORDER if self.facts.get(k)]


# ------------------------------------------------------------------ simulation
class Sim:
    """Owns the world, the props and the cat; builds each frame and its text layer."""

    def __init__(self, W, H, rng, ascii_mode=False, hour_fn=None, coat=None, sysinfo=None):
        self.rng = rng
        self.coat = coat or rng.choice(sorted(COATS))
        self.info = sysinfo or SysInfo()
        self.ascii = ascii_mode
        self.hour_fn = hour_fn
        self.t = 0.0
        self.parts = []
        self.texts = []
        self.cat = None
        self.startle_evt = False
        self.next_startle = rng.uniform(600, 1500)
        self.next_toss = rng.uniform(500, 1200)
        self.panel = True
        self.W = 0
        self.resize(W, H)

    def resize(self, W, H):
        oldW = self.W or W
        self.W, self.H = W, H
        self.S = clamp(min(H / 64, W / 120), 0.55, 4.0)
        self.layout_info()
        self.world = World(W, H, self.S, self.rng, self.ascii, self.info_w + 2 if self.info_w else 0)
        self.world.update(0, self.t, self.hour())
        self.cv = Canvas(W, H)
        if self.cat is None:
            self.laser, self.yarn = Laser(self, self.rng), None
            self.cat = Cat(self, self.rng)
            self.cat.force = ['sit']
        else:
            self.cat.resized(self, W / oldW)
        self.yarn = Yarn(self, self.rng)
        self.mug = Mug(self, self.rng)
        self.laser.on, self.laser.mode = False, None
        self.parts, self.texts = [], []

    def hour(self):
        if self.hour_fn:
            return self.hour_fn(self.t) % 24
        lt = time.localtime()
        return lt.tm_hour + lt.tm_min / 60 + lt.tm_sec / 3600

    def dust(self, x, y, n):
        S = self.S
        for _ in range(n):
            self.parts.append([x + self.rng.uniform(-3, 3) * S, y, self.rng.uniform(-8, 8) * S,
                               -self.rng.uniform(2, 7) * S, 0.0, self.rng.uniform(0.4, 0.8)])

    def say(self, x, y, txt, col):
        self.texts.append([x, y, txt, col, 0.0, 2.4 if txt == 'z' else 1.6])

    def clink(self, x, y):
        self.dust(x, y, 5)
        self.say(x, y - 3 * self.S, '*', (255, 255, 220))

    def step(self, dt):
        self.t += dt
        w = self.world
        w.update(dt, self.t, self.hour())
        self.laser.update(dt, self.t)
        self.yarn.update(dt)
        self.mug.update(dt, self.t)
        self.cat.update(dt)
        self.info.refresh(self.t)
        if self.t > self.next_startle:
            self.startle_evt = True
            self.next_startle = self.t + self.rng.uniform(900, 2400)
        if self.t > self.next_toss:
            if not self.cat.asleep:
                self.yarn.toss()
            self.next_toss = self.t + self.rng.uniform(700, 1800)
        for p in self.parts:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 20 * self.S * dt
            p[4] += dt
        self.parts = [p for p in self.parts if p[4] < p[5]]
        for q in self.texts:
            q[4] += dt
            q[1] -= 3.0 * self.S * dt
            q[0] += math.sin(q[4] * 3) * 1.5 * self.S * dt
        self.texts = [q for q in self.texts if q[4] < q[5]]

    @property
    def bgp(self):
        return 0 if self.ascii else BG_PK

    def next_coat(self):
        names = sorted(COATS)
        self.coat = self.cat.coat = names[(names.index(self.coat) + 1) % len(names)]

    def render(self):
        cv, w, cat, mug = self.cv, self.world, self.cat, self.mug
        cv.px[:] = w.background()
        w.draw_back(cv, self.t)
        mug_behind = mug.state in ('sill', 'fade', 'appear') and cat.level == 'floor'
        if mug_behind:
            mug.draw(cv, w.L, self.ascii)
        cat.draw(cv, (1.0, 1.0, 1.0))
        if not mug_behind:
            mug.draw(cv, w.L, self.ascii)
        self.yarn.draw(cv, (1.0, 1.0, 1.0), self.ascii)
        self.laser.draw(cv, self.ascii)
        dc = (150, 150, 150) if self.ascii else (120, 124, 140)
        for x, y, _, _, age, life in self.parts:
            cv.circle(x, y, 0.6 * self.S, dc, 0, 0.6 * (1 - age / life))
        return cv

    # ---------------------------------------------------------- text layer
    LOGO = ('   /\\_/\\', '  ( {e} )', '   > ^ <', '  /     \\', ' (|     |)', '  (_/ \\_)')

    def layout_info(self):
        """Size the fastfetch pane: logo + facts if there is room, facts only, or nothing."""
        cols, rows = self.W, self.H // 2
        sky_right = 2 + int(clamp(0.24 * cols, 18, 44))
        body = max((len(k) + 2 + len(v) for k, v in self.info.lines()), default=20)
        for logo in (True, False):
            text_w = min(body, 46)
            w = text_w + 4 + (12 if logo else 0)
            if cols - 2 - w > sky_right + 2 and rows >= 16:
                self.info_w, self.info_logo, self.info_text_w = w, logo, text_w
                return
        self.info_w, self.info_logo, self.info_text_w = 0, False, 0

    def overlay(self):
        """Text over the pixels: {row: {col: (ch, fg_sgr, bg_sgr, front)}}. Text that is not 'front'
        only shows where the cell is empty background, so the cat walks in front of the panes."""
        ov = {}
        cols, rows = self.W, self.H // 2
        asc, w, cat = self.ascii, self.world, self.cat
        coat = coat_palette(cat.coat)
        accent = 'orange' if asc else '38;2;%d;%d;%d' % coat['accent']
        dim, txt, faint = ('38;2;88;96;124', '38;2;214;218;230', '38;2;120;126;150')

        def put(cx, cy, s, fg=txt, bg='49', front=False):
            if 0 <= cy < rows:
                row = ov.setdefault(cy, {})
                for i, ch in enumerate(s):
                    if 0 <= cx + i < cols:
                        row[cx + i] = (ch, fg, bg, front)

        def box(c0, r0, c1, r1, title):
            h, v, tl, tr, bl, br = ('-', '|', '+', '+', '+', '+') if asc else ('─', '│', '╭', '╮', '╰', '╯')
            put(c0, r0, tl + h * (c1 - c0 - 1) + tr, dim)
            put(c0, r1, bl + h * (c1 - c0 - 1) + br, dim)
            for r in range(r0 + 1, r1):
                put(c0, r, v, dim)
                put(c1, r, v, dim)
            put(c0 + 2, r0, ' %s ' % title, faint)

        # outside pane (its bottom border is the ledge)
        icon = {'clear': ('*' if asc else '☀') if w.day > 0.5 else ('(' if asc else '☾'),
                'cloudy': '~' if asc else '☁', 'rain': '/' if asc else '☂'}[w.weather]
        box(w.pc0, w.pr0, w.pc1, w.pr1, 'outside %s %s' % (icon, w.weather))
        # date under the tty-clock (and in ASCII mode the clock itself, as text)
        if w.ck and asc:
            lt = time.localtime()
            x = int(w.clock_x - 17 * w.ck / 2)
            for ch in '%02d:%02d' % (lt.tm_hour, lt.tm_min):
                for gy_, row in enumerate(FONT[ch]):
                    for gx_, bit in enumerate(row):
                        if bit == '1' and (ch != ':' or lt.tm_sec % 2 == 0):
                            for yy in range(max(1, w.ck // 2)):
                                put(x + gx_ * w.ck, 1 + gy_ * max(1, w.ck // 2) + yy, '#' * w.ck, txt)
                x += (len(FONT[ch][0]) + 1) * w.ck
        if w.ck:
            d = time.strftime('%A %d %B')
            bottom = 1 + 5 * max(1, w.ck // 2) if asc else math.ceil((2 + 5 * w.ck) / 2)
            put(int(w.clock_x - len(d) / 2), bottom + 1, d, faint)
        # fastfetch pane
        if self.panel and self.info_w:
            c1, r0 = cols - 2, 1
            c0 = c1 - self.info_w + 1
            tw = self.info_text_w
            mood = cat.mood()
            face = '-.-' if cat.asleep else 'O.O' if cat.gen_name in ('hunt', 'zoomies', 'startle') \
                else '^.^' if cat.cur['eye'] < 0.4 else 'o.o'
            n = int(round(clamp(cat.energy, 0, 1) * 6))
            bar = ('#' * n + '-' * (6 - n)) if asc else ('▰' * n + '▱' * (6 - n))
            body = [((self.info.user, accent), ('@', txt), (self.info.host, accent)),
                    (('-' * (len(self.info.user) + len(self.info.host) + 1), txt),)]
            for k, v in self.info.lines():
                v = v if len(k) + 2 + len(v) <= tw else v[:tw - len(k) - 3] + ('~' if asc else '…')
                body.append(((k, accent), (': ' + v, txt)))
            body.append(())
            for k, v in (('Cat', '%s, %s' % (COATS[cat.coat]['name'], COATS[cat.coat]['desc'])), ('Mood', mood),
                         ('Doing', cat.doing), ('Energy', bar)):
                body.append(((k, accent), (': ' + v[:tw - len(k) - 2], txt)))
            if not asc:
                body += [(), 'palette', 'palette2']
            body = body[:max(3, rows - 5)]
            r1 = r0 + len(body) + 1
            box(c0, r0, c1, r1, 'fastfetch')
            x0 = c0 + 2
            if self.info_logo:
                for i, l in enumerate(self.LOGO):
                    put(x0, r0 + 1 + i, l.format(e=face), accent)
                x0 += 12
            for i, line in enumerate(body):
                y = r0 + 1 + i
                if line in ('palette', 'palette2'):
                    base = 40 if line == 'palette' else 100
                    for j in range(8):
                        put(x0 + j * 3, y, '   ' if not asc else '###', '39' if not asc else txt,
                            '%d' % (base + j))
                    continue
                x = x0
                for seg, fg in line:
                    put(x, y, seg, fg)
                    x += len(seg)
            put(c0 + 2, r1, ' q quit · l laser · y yarn · c coat · i info ' if not asc
                else ' q quit  l laser  y yarn  c coat  i info ', faint)
        # status bar: the floor
        y = rows - 1
        name = COATS[cat.coat]['name'].lower()
        left = [(' %s ' % name, '38;2;20;20;28', '48;2;%d;%d;%d' % coat['accent']),
                (' %s ' % cat.mood(), txt, '48;2;58;62;84'), (' %s ' % cat.doing, txt, '48;2;40;43;58')]
        info = self.info
        right = [(' load %.2f ' % info.load[0], txt, '48;2;40;43;58'),
                 (' mem %d%% ' % info.mem_pct, txt, '48;2;58;62;84'),
                 (time.strftime(' %a %d %b '), txt, '48;2;40;43;58'),
                 (time.strftime(' %H:%M '), '38;2;20;20;28', '48;2;%d;%d;%d' % coat['accent'])]
        put(0, y, ' ' * cols, txt, '48;2;28;30;40', True)
        x = 0
        for seg, fg, bg in left:
            put(x, y, seg, fg, bg, True)
            x += len(seg) + (1 if asc else 0)
        x = cols - sum(len(seg) for seg, _, _ in right)
        for seg, fg, bg in right:
            put(x, y, seg, fg, bg, True)
            x += len(seg)
        # floating glyphs (z, !, ?, hearts) are always in front
        for x_, y_, t_, col, age, life in self.texts:
            if t_ == 'z' and age > 1.0:
                t_ = 'Z'
            if t_ == '<3' and not asc:
                t_ = '♥'
            c = mulc(col, clamp((life - age) * 2, 0.2, 1))
            put(int(x_), int(y_ / 2), t_, '38;2;%d;%d;%d' % tuple(int(v) for v in c), '49', True)
        return ov


# ------------------------------------------------------------------ terminal output
RAMP = ((55, ':'), (85, ':'), (115, '='), (140, '+'), (170, '*'), (200, '#'), (230, '%'), (256, '@'))


class Screen:
    """Frame encoder: diffs against the last frame so each frame is one small write."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = self.prevpx = None
        self.prev_ov = {}

    def encode(self, px, W, rows, ov, ascii_mode, bgp=BG_PK):
        """Diff against the last frame and emit one string. Pixels equal to bgp stay
        transparent (terminal default background)."""
        n = W * rows
        if self.prev is None or len(self.prev) != n:
            self.prev = [None] * n
            self.prevpx = None
        prev, ppx, pov = self.prev, self.prevpx, self.prev_ov
        out = []
        ap = out.append
        fg = bg = None
        last = -2
        ink = [lum(c) > 30 for c in px] if ascii_mode else None
        for cy in range(rows):
            base = cy * 2 * W
            orow = ov.get(cy)
            if ppx is not None and orow == pov.get(cy) and px[base:base + 2 * W] == ppx[base:base + 2 * W]:
                continue
            ri = cy * W
            for cx in range(W):
                t, b = px[base + cx], px[base + W + cx]
                cell = orow.get(cx) if orow else None
                if cell is not None and (cell[3] or (t == bgp and b == bgp)):
                    key = cell[:3]
                elif ascii_mode:
                    it, ib = ink[base + cx], ink[base + W + cx]
                    if not it and not ib:
                        key = ' '
                    elif it and not ib:
                        run = 0 < cx < W - 1 and ink[base + cx - 1] and ink[base + cx + 1]
                        key = '-' if run else "'"
                    elif ib and not it:
                        run = 0 < cx < W - 1 and ink[base + cx - 1] and ink[base + cx + 1]
                        key = '_' if run else '.'
                    elif (cx == 0 or not ink[base + cx - 1]) and (cx == W - 1 or not ink[base + cx + 1]):
                        key = '|'
                    else:
                        L = (lum(t) + lum(b)) / 2
                        key = next(ch for th, ch in RAMP if L < th)
                else:
                    key = (t << 24) | b
                i = ri + cx
                if prev[i] == key:
                    continue
                prev[i] = key
                if i != last + 1 or cx == 0:  # autowrap is off: never rely on wrapping
                    ap('\x1b[%d;%dH' % (cy + 1, cx + 1))
                last = i
                if ascii_mode:
                    ap(key if type(key) is str else key[0])
                    continue
                if type(key) is int:
                    if t == bgp:
                        if b == bgp:
                            nf, nb, ch = fg, '49', ' '
                        else:
                            nf, nb, ch = '38;2;%d;%d;%d' % (b >> 16, (b >> 8) & 255, b & 255), '49', '▄'
                    elif b == bgp:
                        nf, nb, ch = '38;2;%d;%d;%d' % (t >> 16, (t >> 8) & 255, t & 255), '49', '▀'
                    elif t == b:
                        nf, nb, ch = fg, '48;2;%d;%d;%d' % (t >> 16, (t >> 8) & 255, t & 255), ' '
                    else:
                        nf = '38;2;%d;%d;%d' % (t >> 16, (t >> 8) & 255, t & 255)
                        nb, ch = '48;2;%d;%d;%d' % (b >> 16, (b >> 8) & 255, b & 255), '▀'
                else:
                    ch, nf, nb = key
                if nf != fg:
                    ap('\x1b[%sm' % nf)
                    fg = nf
                if nb != bg:
                    ap('\x1b[%sm' % nb)
                    bg = nb
                ap(ch)
        self.prevpx = px[:]
        self.prev_ov = ov
        return ''.join(out)


def save_png(path, px, w, h, scale=4):
    raw = bytearray()
    for y in range(h):
        line = bytearray()
        for c in px[y * w:(y + 1) * w]:
            line += bytes((c >> 16, (c >> 8) & 255, c & 255)) * scale
        for _ in range(scale):
            raw += b'\x00' + line

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w * scale, h * scale, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(bytes(raw), 6)) + chunk(b'IEND', b''))


# ------------------------------------------------------------------ entry points
def parse_input(data, sim):
    """Keys and SGR mouse reports. Returns False to quit."""
    i = 0
    while i < len(data):
        if data.startswith('\x1b[<', i):
            j = i + 3
            while j < len(data) and data[j] not in 'Mm':
                j += 1
            try:
                b, x, y = (int(v) for v in data[i + 3:j].split(';'))
                sim.laser.mouse_t = sim.t
                sim.laser.mx, sim.laser.my = x - 0.5, (y - 1) * 2 + 1.0
            except ValueError:
                pass
            i = j + 1
            continue
        ch = data[i]
        if ch in 'qQ':
            return False
        if ch == 'i':
            sim.panel = not sim.panel
        elif ch == 'l':
            if sim.laser.mode == 'ai':
                sim.laser.stop(sim.t)
            else:
                sim.laser.start(sim.t)
        elif ch == 'y':
            sim.yarn.toss()
        elif ch == 'c':
            sim.next_coat()
        elif ch == 'a':
            sim.ascii = not sim.ascii
            sim.resize(sim.W, sim.H)
        i += 1
    return True


def sky_clock(args):
    """Hour-of-day source for the sky: real time unless --hour / --timelapse say otherwise."""
    if args.hour is None and args.timelapse == 1:
        return None
    lt = time.localtime()
    h0 = args.hour if args.hour is not None else lt.tm_hour + lt.tm_min / 60
    return lambda t: h0 + t * args.timelapse / 3600


def run(args):
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    out = sys.stdout
    enter = '\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J' + ('\x1b[?1003h\x1b[?1006h' if args.mouse else '')
    leave = '\x1b[?1003l\x1b[?1006l\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l'
    hour_fn = sky_clock(args)
    try:
        tty.setcbreak(fd)
        out.write(enter)
        out.flush()
        cols, lines = os.get_terminal_size()
        sim = Sim(cols, lines * 2, random.Random(args.seed), args.ascii, hour_fn, args.coat)
        if args.force:
            sim.cat.force = args.force.split(',')
        scr = Screen()
        frame = 1.0 / args.fps
        last = time.monotonic()
        while True:
            now = time.monotonic()
            dt, last = min(0.1, now - last), now
            cols, lines = os.get_terminal_size()
            if (cols, lines * 2) != (sim.W, sim.H):
                sim.resize(cols, lines * 2)
                scr.reset()
                out.write('\x1b[0m\x1b[2J')
            asc = sim.ascii
            sim.step(dt)
            cv = sim.render()
            s = scr.encode(cv.px, sim.W, sim.H // 2, sim.overlay(), sim.ascii, sim.bgp)
            if asc != sim.ascii:
                scr.reset()
            if s:
                out.write('\x1b[?2026h' + s + '\x1b[?2026l')
                out.flush()
            # asleep and nothing going on: 10 fps is plenty for slow breathing
            calm = sim.cat.asleep and not sim.laser.on and sim.mug.state != 'fall' and sim.world.bird.state == 'gone'
            deadline = now + (max(frame, 0.1) if calm else frame)
            while True:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                r, _, _ = select.select([fd], [], [], rem)
                if r:
                    data = os.read(fd, 4096).decode('utf-8', 'ignore')
                    if not parse_input(data, sim):
                        return
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        out.write(leave)
        out.flush()


def soak(args):
    """Headless run: several seeds, random laser/yarn/mouse pokes, invariants + timing."""
    worst = 0.0
    for seed in range(args.seeds):
        rng = random.Random(seed)
        W, H = rng.choice(((80, 48), (120, 70), (200, 110), (60, 30), (40, 24)))
        sim = Sim(W, H, rng, seed % 4 == 3, lambda t: 6 + t / 120)
        scr = Screen()
        dt, n = 1 / 20, int(args.soak * 20)
        counts, longest, cur, cur_t0 = {}, {}, None, 0.0
        prev_head, jumps, prev_w = None, 0, 0
        times = []
        for i in range(n):
            if rng.random() < 1 / 1500:
                sim.laser.start(sim.t)
            if rng.random() < 1 / 4000:
                sim.yarn.toss()
            if rng.random() < 1 / 6000:  # resize mid-run
                sim.resize(*rng.choice(((80, 48), (120, 70), (100, 60))))
                scr.reset()
            t0 = time.perf_counter()
            sim.step(dt)
            cv = sim.render()
            scr.encode(cv.px, sim.W, sim.H // 2, sim.overlay(), sim.ascii, sim.bgp)
            times.append(time.perf_counter() - t0)
            c = sim.cat
            lo, hi = sim.world.bounds(c.level)
            assert all(math.isfinite(v) for v in (c.x, c.y, c.th, c.v)), 'non-finite state'
            assert lo - 4 * sim.S <= c.x <= hi + 4 * sim.S or c.airborne, ('out of bounds', c.x, lo, hi, c.gen_name)
            assert 0 <= sim.yarn.x <= sim.W
            hp = c.head_pos()
            # a head moving > 4 units in one frame outside jumps/zoomies would be a visible snap
            fast = prev_head and math.hypot(hp[0] - prev_head[0], hp[1] - prev_head[1]) > 4.0 * sim.S
            if fast and sim.W == prev_w and not c.airborne and c.gen_name != 'zoomies':
                jumps += 1
            prev_head, prev_w = hp, sim.W
            if c.gen_name != cur or c.gen_t0 != cur_t0:
                cur, cur_t0 = c.gen_name, c.gen_t0
                counts[cur] = counts.get(cur, 0) + 1
            longest[cur] = max(longest.get(cur, 0), sim.t - cur_t0)
        times.sort()
        worst = max(worst, times[-1])
        print('seed %d %dx%d %s: mean %.1fms p95 %.1fms max %.1fms' % (
            seed, W, H, 'ascii' if seed % 4 == 3 else 'color', 1000 * sum(times) / len(times),
            1000 * times[int(len(times) * 0.95)], 1000 * times[-1]))
        print('   counts ', dict(sorted(counts.items())))
        print('   longest', {k: round(v) for k, v in sorted(longest.items())}, 'snaps', jumps)
        bad = {k: v for k, v in longest.items() if v > (700 if k in ('nap', 'loaf', 'window') else 240)}
        assert not bad, ('behaviour ran too long', bad)
    print('ok')


def shots(args):
    """Render the pixel layer of a headless run to PNGs, for eyeballing the rig."""
    os.makedirs(args.shots, exist_ok=True)
    sim = Sim(args.size[0], args.size[1], random.Random(args.seed), args.ascii, sky_clock(args), args.coat)
    if args.force:
        sim.cat.force = args.force.split(',')
        sim.cat.switch(sim.cat.force.pop(0))
    if args.laser:
        sim.laser.start(0.5, 200)
    k = 0
    for i in range(args.frames):
        sim.step(1 / 20)
        if i % args.every == 0:
            cv = sim.render()
            save_png(os.path.join(args.shots, 'f%04d.png' % k), cv.px, sim.W, sim.H, args.scale)
            print(k, round(sim.t, 2), sim.cat.gen_name, sim.cat.doing, sim.cat.posture)
            k += 1


def main():
    ap = argparse.ArgumentParser(description='A cat lives in your idle terminal.')
    ap.add_argument('--ascii', action='store_true', help='plain ASCII mode, no colour')
    ap.add_argument('--coat', choices=sorted(COATS), default=None, help='pick the cat (default: random)')
    ap.add_argument('--fps', type=float, default=20, help='frame rate while awake (default 20)')
    ap.add_argument('--no-mouse', dest='mouse', action='store_false', help="don't capture the mouse")
    ap.add_argument('--hour', type=float, default=None, help='pretend it is this hour (0-24)')
    ap.add_argument('--timelapse', type=float, default=1.0, help='speed up the day/night cycle')
    ap.add_argument('--seed', type=int, default=None, help='random seed, for reproducible runs')
    dev = ap.add_argument_group('development')
    dev.add_argument('--force', default='', help='comma-separated behaviours to run first, e.g. zoomies,knock')
    dev.add_argument('--soak', type=float, default=0, help='headless soak test: seconds of sim per seed')
    dev.add_argument('--seeds', type=int, default=4, help='seeds for --soak')
    dev.add_argument('--shots', default=None, help='dump the pixel layer of a headless run as PNGs here')
    dev.add_argument('--frames', type=int, default=200, help='frames to simulate for --shots')
    dev.add_argument('--every', type=int, default=10, help='save every Nth frame for --shots')
    dev.add_argument('--scale', type=int, default=4, help='PNG upscale for --shots')
    dev.add_argument('--size', type=int, nargs=2, default=(120, 70), help='pixel size W H for --shots')
    dev.add_argument('--laser', action='store_true', help='start a laser session for --shots')
    args = ap.parse_args()
    if args.soak:
        soak(args)
    elif args.shots:
        shots(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
