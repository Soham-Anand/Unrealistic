#!/usr/bin/env python3
"""Generate synthetic math data for Prabhakar (Phase 5 Math Specialization).

Each output format is a standalone generator that writes its own bin:
  - arithmetic      direct computation (add/sub/mul/div/sqrt/pow/mod) + worked
  - cot_scratchpad  digit-by-digit chain-of-thought (long mul/div, column add, powers)
  - arithmetic_worked worked multi-step arithmetic (solve, fractions, %, decimals)
  - problem          word problems with worked solutions & verification
  - algebra         equation solving, factor, inequality
  - advanced_algebra polynomials, logs, exp, abs value, determinants
  - number_theory   primes, divisibility, gcd/lcm, mod, digits
  - geometry        area, perimeter, volume, Pythagoras, angles
  - trig            sin/cos/tan standard angles, identities
  - combinatorics   permutations, combinations, probability
  - verification    candidate solution -> verify (valid & invalid)
  - error           identify the mistake
  - simplify        expression simplification
  - code            code-as-solver (Python)
  - clean           clean-completion Q&A (full-sentence responses)

Generators are independent of curriculum; the Phase 5 curriculum builder
(build_phase5_curriculum.py) reads these bins and assembles stages.

Usage:
  python3 scripts/gen_math_data.py --all --smoke 10000   (validate, small)
  python3 scripts/gen_math_data.py --validate             (quality-check only)
  python3 scripts/gen_math_data.py --all --tokens 100000000

Output: data/phase5/*_syn.bin, one per generator.
"""

import os, sys, random, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase5"
TOKENIZER_PATH = "data/tokenizer/phase1.model"
EOS = None

# ─── Difficulty 1: Arithmetic ──────────────────────────────────────

def gen_arithmetic_direct(rng, n):
    """Level 1: Direct computation — addition, subtraction, mul, div, roots, powers."""
    templates = []
    for _ in range(n):
        op = rng.choice(["add", "sub", "mul", "div", "sqrt", "pow", "mod"])
        if op == "add":
            a, b = rng.randint(0, 999), rng.randint(0, 999)
            ans = a + b
            fmt = rng.choice([
                f"{a} + {b} = {ans}",
                f"What is {a} + {b}? The answer is {ans}.",
                f"{a} plus {b} equals {ans}.",
                f"Calculate {a} + {b}. Solution: {ans}",
            ])
        elif op == "sub":
            a = rng.randint(10, 999)
            b = rng.randint(0, a)
            ans = a - b
            fmt = rng.choice([
                f"{a} - {b} = {ans}",
                f"What is {a} minus {b}? The answer is {ans}.",
                f"{a} subtract {b} equals {ans}.",
                f"Calculate {a} - {b}. Solution: {ans}",
            ])
        elif op == "mul":
            a, b = rng.randint(0, 30), rng.randint(0, 30)
            ans = a * b
            fmt = rng.choice([
                f"{a} * {b} = {ans}",
                f"What is {a} times {b}? The answer is {ans}.",
                f"{a} multiplied by {b} equals {ans}.",
                f"Calculate {a} x {b}. Solution: {ans}",
            ])
        elif op == "div":
            b = rng.randint(1, 30)
            q = rng.randint(0, 30)
            a = b * q
            ans = q
            fmt = rng.choice([
                f"{a} / {b} = {ans}",
                f"What is {a} divided by {b}? The answer is {ans}.",
                f"{a} divided by {b} equals {ans}.",
                f"Calculate {a} / {b}. Solution: {ans}",
            ])
        elif op == "sqrt":
            r = rng.randint(1, 25)
            sq = r * r
            fmt = rng.choice([
                f"The square root of {sq} is {r}.",
                f"What is sqrt({sq})? The answer is {r}.",
                f"sqrt({sq}) = {r}",
            ])
        elif op == "pow":
            base = rng.randint(2, 10)
            exp = rng.randint(2, 5)
            ans = base ** exp
            fmt = rng.choice([
                f"{base}^{exp} = {ans}",
                f"What is {base} raised to the power of {exp}? The answer is {ans}.",
                f"{base} to the power {exp} equals {ans}.",
            ])
        elif op == "mod":
            b = rng.randint(2, 20)
            a = rng.randint(b, 100)
            ans = a % b
            fmt = rng.choice([
                f"{a} mod {b} = {ans}",
                f"What is {a} modulo {b}? The answer is {ans}.",
                f"The remainder of {a} / {b} is {ans}.",
            ])
        templates.append(fmt)
    return templates


def gen_arithmetic_worked(rng, n):
    """Level 1-2: Worked solutions with step-by-step reasoning."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["two_step", "multi_op", "fraction", "percent", "decimal"])
        if kind == "two_step":
            a, b, c = rng.randint(1, 50), rng.randint(1, 50), rng.randint(1, 20)
            # a*x + b = c  => x = (c-b)/a but we want clean answers
            x = rng.randint(1, 10)
            left_const = rng.randint(1, 20)
            result = a * x + left_const
            fmt = (
                f"Solve: {a}x + {left_const} = {result}\n"
                f"Step 1: Subtract {left_const} from both sides: {a}x = {result - left_const}\n"
                f"Step 2: Divide both sides by {a}: x = {(result - left_const) // a}\n"
                f"Verification: {a} * {(result - left_const) // a} + {left_const} = {a * ((result - left_const) // a) + left_const} ✓\n"
                f"Answer: x = {(result - left_const) // a}"
            )
        elif kind == "multi_op":
            a, b, c = rng.randint(2, 15), rng.randint(2, 15), rng.randint(1, 10)
            d = rng.randint(1, 20)
            result = a * b + c * d
            fmt = (
                f"Calculate: {a} × {b} + {c} × {d}\n"
                f"Step 1: {a} × {b} = {a * b}\n"
                f"Step 2: {c} × {d} = {c * d}\n"
                f"Step 3: {a * b} + {c * d} = {result}\n"
                f"Answer: {result}"
            )
        elif kind == "fraction":
            num1, den1 = rng.randint(1, 10), rng.randint(2, 12)
            num2, den2 = rng.randint(1, 10), rng.randint(2, 12)
            common = den1 * den2
            new_num1 = num1 * den2
            new_num2 = num2 * den1
            total = new_num1 + new_num2
            fmt = (
                f"Add: {num1}/{den1} + {num2}/{den2}\n"
                f"Step 1: Common denominator = {den1} × {den2} = {common}\n"
                f"Step 2: {num1}/{den1} = {new_num1}/{common}\n"
                f"Step 3: {num2}/{den2} = {new_num2}/{common}\n"
                f"Step 4: {new_num1}/{common} + {new_num2}/{common} = {total}/{common}\n"
                f"Answer: {total}/{common}"
            )
        elif kind == "percent":
            whole = rng.choice([50, 100, 200, 250, 500])
            pct = rng.choice([10, 15, 20, 25, 30, 50])
            part = whole * pct // 100
            fmt = (
                f"What is {pct}% of {whole}?\n"
                f"Step 1: {pct}% = {pct}/100 = {pct/100}\n"
                f"Step 2: {pct/100} × {whole} = {part}\n"
                f"Answer: {part}"
            )
        elif kind == "decimal":
            a = round(rng.uniform(1, 20), 1)
            b = round(rng.uniform(1, 20), 1)
            ans = round(a + b, 1)
            fmt = (
                f"Add: {a} + {b}\n"
                f"Step 1: Align decimal points\n"
                f"Step 2: {a} + {b} = {ans}\n"
                f"Answer: {ans}"
            )
        templates.append(fmt)
    return templates


def gen_cot_scratchpad(rng, n):
    """Level 2-3: True digit-by-digit chain-of-thought scratchpads.

    Teaches the model to COMPUTE step-by-step (carry/borrow/partial
    products) rather than jump straight to a memorized answer. Long
    multiplication, long division (with remainder), and column addition.
    This is the 'actual thinking' mechanism (scratchpad reasoning).
    """
    templates = []
    for _ in range(n):
        kind = rng.choice(["long_mul", "long_div", "col_add", "multi_pow"])
        if kind == "long_mul":
            a = rng.randint(12, 99)
            b = rng.randint(11, 99)
            prod = a * b
            # partial products: a * ones, then a * tens
            ones = b % 10
            tens = b // 10
            p1 = a * ones
            p2 = a * tens
            fmt = (
                f"Multiply: {a} × {b}\n"
                f"Step 1: split {b} = {tens} tens + {ones} ones\n"
                f"Step 2: {a} × {ones} = {p1}\n"
                f"Step 3: {a} × {tens} = {p2} (write as {p2}0)\n"
                f"Step 4: partial products {p1} + {p2}0 = {prod}\n"
                f"Answer: {prod}"
            )
        elif kind == "long_div":
            q = rng.randint(2, 20)
            div = rng.randint(2, 12)
            dividend = q * div
            r = rng.randint(0, div - 1) if rng.random() < 0.3 else 0
            d2 = dividend + r
            if r == 0:
                fmt = (
                    f"Divide: {d2} ÷ {div}\n"
                    f"Step 1: {div} × {q} = {dividend}\n"
                    f"Step 2: {dividend} = {d2}, remainder = {d2} - {dividend} = 0\n"
                    f"Answer: {q}"
                )
            else:
                fmt = (
                    f"Divide: {d2} ÷ {div}\n"
                    f"Step 1: {div} × {q} = {dividend}\n"
                    f"Step 2: remainder = {d2} - {dividend} = {r}\n"
                    f"Answer: {q} remainder {r}"
                )
        elif kind == "col_add":
            a = rng.randint(100, 999)
            b = rng.randint(100, 999)
            s = a + b
            a_s, b_s = str(a), str(b)
            # column carry demo (units, tens, hundreds)
            units = int(a_s[-1]) + int(b_s[-1])
            tens_p = int(a_s[-2]) + int(b_s[-2]) + (units // 10)
            hun_p = int(a_s[0]) + int(b_s[0]) + (tens_p // 10)
            fmt = (
                f"Add: {a} + {b}\n"
                f"Step 1: units: {a_s[-1]} + {b_s[-1]} = {units} (write {units % 10}, carry {units // 10})\n"
                f"Step 2: tens: {a_s[-2]} + {b_s[-2]} + carry = {tens_p} (write {tens_p % 10}, carry {tens_p // 10})\n"
                f"Step 3: hundreds: {a_s[0]} + {b_s[0]} + carry = {hun_p}\n"
                f"Answer: {s}"
            )
        elif kind == "multi_pow":
            base = rng.randint(2, 9)
            exp = rng.randint(3, 6)
            val = base ** exp
            fmt = (
                f"Compute: {base}^{exp}\n"
                f"Step 1: {base} × {base} = {base * base}\n"
                f"Step 2: {base * base} × {base} = {base ** 3}\n"
                + (f"Step 3: {base ** 3} × {base} = {base ** 4}\n" if exp >= 4 else "")
                + (f"Step 4: {base ** 4} × {base} = {base ** 5}\n" if exp >= 5 else "")
                + (f"Step 5: {base ** 5} × {base} = {base ** 6}\n" if exp >= 6 else "")
                + f"Answer: {val}"
            )
        templates.append(fmt)
    return templates


# ─── Difficulty 2-3: Algebra ───────────────────────────────────────

def gen_algebra_direct(rng, n):
    """Level 2-3: Equation solving, simplification, factoring."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["one_step", "two_step", "multi_step", "quadratic",
                           "system", "simplify", "factor", "inequality"])
        if kind == "one_step":
            a = rng.randint(2, 15)
            b = rng.randint(1, 50)
            x = rng.randint(1, 20)
            result = a * x
            fmt = rng.choice([
                f"{a}x = {result}. Solution: x = {x}",
                f"Solve {a}x = {result}. x = {x}",
                f"Find x: {a}x = {result}. Answer: x = {x}",
            ])
        elif kind == "two_step":
            a = rng.randint(2, 10)
            b = rng.randint(1, 30)
            x = rng.randint(1, 15)
            result = a * x + b
            fmt = rng.choice([
                f"Solve {a}x + {b} = {result}. x = {x}",
                f"{a}x + {b} = {result}. Find x. Answer: x = {x}",
                f"Solve for x: {a}x + {b} = {result}. x = {x}",
            ])
        elif kind == "multi_step":
            a = rng.randint(2, 8)
            b = rng.randint(1, 15)
            c = rng.randint(1, 10)
            x = rng.randint(1, 10)
            result = a * (x + c) + b
            fmt = (
                f"Solve: {a}(x + {c}) + {b} = {result}\n"
                f"Step 1: {a}(x + {c}) = {result - b}\n"
                f"Step 2: x + {c} = {(result - b) // a}\n"
                f"Step 3: x = {(result - b) // a - c}\n"
                f"Answer: x = {(result - b) // a - c}"
            )
        elif kind == "quadratic":
            # x^2 + bx + c = 0 with integer roots
            r1 = rng.randint(-8, 8)
            r2 = rng.randint(-8, 8)
            b_coef = -(r1 + r2)
            c_coef = r1 * r2
            fmt = rng.choice([
                f"Solve x^2 + ({b_coef})x + ({c_coef}) = 0. Factor: (x - {r1})(x - {r2}) = 0. Solutions: x = {r1} or x = {r2}",
                f"Factor x^2 + ({b_coef})x + ({c_coef}). Answer: (x - {r1})(x - {r2}). Roots: {r1}, {r2}",
                f"Find roots of x^2 + ({b_coef})x + ({c_coef}) = 0. x = {r1} or x = {r2}",
            ])
        elif kind == "system":
            x = rng.randint(1, 10)
            y = rng.randint(1, 10)
            a1, b1 = rng.randint(1, 5), rng.randint(1, 5)
            a2, b2 = rng.randint(1, 5), rng.randint(1, 5)
            # Ensure unique solution
            while a1 * b2 == a2 * b1:
                a2, b2 = rng.randint(1, 5), rng.randint(1, 5)
            c1 = a1 * x + b1 * y
            c2 = a2 * x + b2 * y
            fmt = (
                f"Solve the system: {a1}x + {b1}y = {c1} and {a2}x + {b2}y = {c2}\n"
                f"Solution: x = {x}, y = {y}\n"
                f"Verification: {a1}({x}) + {b1}({y}) = {c1} ✓ and {a2}({x}) + {b2}({y}) = {c2} ✓"
            )
        elif kind == "simplify":
            a = rng.randint(1, 10)
            b = rng.randint(1, 20)
            c = rng.randint(1, 10)
            d = rng.randint(1, 20)
            coeff = a + c
            const = b + d
            fmt = rng.choice([
                f"Simplify: {a}x + {b} + {c}x + {d} = {coeff}x + {const}",
                f"Combine like terms: {a}x + {b} + {c}x + {d}. Answer: {coeff}x + {const}",
                f"{a}x + {b} + {c}x + {d} = ? Answer: {coeff}x + {const}",
            ])
        elif kind == "factor":
            a = rng.randint(2, 8)
            b = rng.randint(1, 12)
            # Factor a*x^2 + a*b*x = a*x(x + b)
            fmt = rng.choice([
                f"Factor: {a}x^2 + {a * b}x. Answer: {a}x(x + {b})",
                f"Factor out the GCF: {a}x^2 + {a * b}x. Answer: {a}x(x + {b})",
            ])
        elif kind == "inequality":
            a = rng.randint(1, 10)
            x = rng.randint(1, 20)
            b = rng.randint(1, 50)
            result = a * x
            fmt = rng.choice([
                f"Solve: {a}x < {result + a}. Answer: x < {x + 1}",
                f"Find x: {a}x <= {result}. Answer: x <= {x}",
                f"Solve the inequality: {a}x > {result - a}. Answer: x > {x - 1}",
            ])
        templates.append(fmt)
    return templates


# ─── Difficulty 3-4: Problem Solving ───────────────────────────────

def gen_problem_solving(rng, n):
    """Level 3-4: Word problems with worked solutions and verification."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["rate", "mixture", "profit", "age", "distance",
                           "work", "sequence", "combinatorics", "probability"])
        if kind == "rate":
            speed = rng.randint(30, 80)
            time = rng.randint(2, 6)
            dist = speed * time
            fmt = (
                f"A car travels at {speed} km/h for {time} hours. How far does it go?\n"
                f"Step 1: Distance = Speed × Time\n"
                f"Step 2: Distance = {speed} × {time} = {dist} km\n"
                f"Verification: {speed} × {time} = {dist} ✓\n"
                f"Answer: {dist} km"
            )
        elif kind == "mixture":
            price1 = rng.randint(2, 8)
            qty1 = rng.randint(1, 10)
            price2 = rng.randint(2, 8)
            qty2 = rng.randint(1, 10)
            total_cost = price1 * qty1 + price2 * qty2
            total_qty = qty1 + qty2
            avg = total_cost / total_qty
            fmt = (
                f"Mix {qty1} kg at ${price1}/kg with {qty2} kg at ${price2}/kg. What is the average price?\n"
                f"Step 1: Cost of first = {qty1} × ${price1} = ${qty1 * price1}\n"
                f"Step 2: Cost of second = {qty2} × ${price2} = ${qty2 * price2}\n"
                f"Step 3: Total cost = ${total_cost}, Total weight = {total_qty} kg\n"
                f"Step 4: Average = ${total_cost}/{total_qty} = ${avg:.2f}/kg\n"
                f"Answer: ${avg:.2f}/kg"
            )
        elif kind == "profit":
            cost = rng.randint(10, 100)
            pct = rng.choice([10, 15, 20, 25, 30, 50])
            profit = cost * pct // 100
            sell = cost + profit
            fmt = (
                f"A product costs ${cost}. Sold at {pct}% profit. What is the selling price?\n"
                f"Step 1: Profit = {pct}% of ${cost} = ${profit}\n"
                f"Step 2: Selling price = ${cost} + ${profit} = ${sell}\n"
                f"Answer: ${sell}"
            )
        elif kind == "age":
            age1 = rng.randint(10, 40)
            diff = rng.randint(2, 15)
            age2 = age1 + diff
            years = rng.randint(2, 10)
            fmt = (
                f"Person A is {age1} years old. Person B is {diff} years older. "
                f"In {years} years, what will be their combined age?\n"
                f"Step 1: Person B is {age2} years old\n"
                f"Step 2: In {years} years: A = {age1 + years}, B = {age2 + years}\n"
                f"Step 3: Combined = {age1 + years} + {age2 + years} = {age1 + age2 + 2 * years}\n"
                f"Answer: {age1 + age2 + 2 * years}"
            )
        elif kind == "distance":
            speed1 = rng.randint(30, 60)
            speed2 = rng.randint(30, 60)
            time = rng.randint(1, 5)
            total = (speed1 + speed2) * time
            fmt = (
                f"Two trains move toward each other at {speed1} and {speed2} km/h. "
                f"After {time} hours they meet. How far apart were they?\n"
                f"Step 1: Combined speed = {speed1} + {speed2} = {speed1 + speed2} km/h\n"
                f"Step 2: Distance = {speed1 + speed2} × {time} = {total} km\n"
                f"Answer: {total} km"
            )
        elif kind == "work":
            rate1 = rng.choice([2, 3, 4, 5, 6])
            rate2 = rng.choice([2, 3, 4, 5, 6])
            combined = rate1 + rate2
            total_work = 1  # 1 job
            hours_frac = total_work / combined
            fmt = (
                f"Worker A does {rate1} jobs/hour, Worker B does {rate2} jobs/hour. "
                f"How long to complete 1 job together?\n"
                f"Step 1: Combined rate = {rate1} + {rate2} = {combined} jobs/hour\n"
                f"Step 2: Time = 1/{combined} hour = {hours_frac:.4f} hours\n"
                f"Answer: 1/{combined} hour"
            )
        elif kind == "sequence":
            start = rng.randint(1, 10)
            diff = rng.randint(1, 10)
            n_terms = rng.randint(5, 15)
            last = start + (n_terms - 1) * diff
            total = n_terms * (start + last) // 2
            fmt = (
                f"Arithmetic sequence: {start}, {start + diff}, {start + 2*diff}, ... "
                f"Find the sum of the first {n_terms} terms.\n"
                f"Step 1: Last term = {start} + ({n_terms} - 1) × {diff} = {last}\n"
                f"Step 2: Sum = {n_terms} × ({start} + {last}) / 2 = {total}\n"
                f"Answer: {total}"
            )
        elif kind == "combinatorics":
            n = rng.randint(5, 15)
            r = rng.randint(2, min(n - 1, 5))
            result = math.comb(n, r)
            fmt = (
                f"How many ways to choose {r} items from {n}?\n"
                f"Step 1: C({n}, {r}) = {n}! / ({r}! × {n - r}!)\n"
                f"Step 2: C({n}, {r}) = {result}\n"
                f"Answer: {result}"
            )
        elif kind == "probability":
            total = rng.randint(5, 20)
            favorable = rng.randint(1, total - 1)
            from math import gcd
            g = gcd(favorable, total)
            num, den = favorable // g, total // g
            fmt = (
                f"A bag has {total} balls: {favorable} red and {total - favorable} blue. "
                f"What is the probability of drawing red?\n"
                f"Step 1: P(red) = {favorable}/{total}\n"
                f"Step 2: Simplify = {num}/{den}\n"
                f"Answer: {num}/{den}"
            )
        templates.append(fmt)
    return templates


# ─── Error Identification ──────────────────────────────────────────

def gen_error_identification(rng, n):
    """Incorrect solutions → find the error."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["sign_error", "order_ops", "wrong_op", "factoring",
                           "distribution", "division", "fraction"])
        if kind == "sign_error":
            a = rng.randint(2, 10)
            b = rng.randint(1, 20)
            x = rng.randint(1, 10)
            result = a * x + b
            wrong_result = a * x - b  # sign error
            fmt = (
                f"Student solution for {a}x + {b} = {result}:\n"
                f"{a}x = {result} + {b}\n"
                f"{a}x = {wrong_result}\n"
                f"x = {wrong_result // a}\n\n"
                f"Find the error.\n\n"
                f"The error is in Step 1. When moving +{b} to the other side, "
                f"it should be subtracted, not added.\n"
                f"Correct: {a}x = {result} - {b} = {result - b}\n"
                f"x = {(result - b) // a}"
            )
        elif kind == "order_ops":
            a, b, c = rng.randint(2, 8), rng.randint(1, 10), rng.randint(1, 10)
            wrong = (a + b) * c
            correct = a + b * c
            fmt = (
                f"Student solution for {a} + {b} × {c}:\n"
                f"Step 1: {a} + {b} = {a + b}\n"
                f"Step 2: {a + b} × {c} = {wrong}\n\n"
                f"Find the error.\n\n"
                f"The error is in Step 1. Multiplication must be done before addition.\n"
                f"Correct: {b} × {c} = {b * c}, then {a} + {b * c} = {correct}"
            )
        elif kind == "wrong_op":
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            fmt = (
                f"Student solution for {a}² + {b}²:\n"
                f"{a}² = {a * a}\n"
                f"{b}² = {b * b}\n"
                f"{a}² + {b}² = {a * a + b * b}\n\n"
                f"But the student wrote {a}² + {b}² = ({a + b})² = {(a + b) ** 2}.\n"
                f"Find the error.\n\n"
                f"The error is: (a + b)² ≠ a² + b². The correct formula is (a + b)² = a² + 2ab + b².\n"
                f"The student incorrectly distributed the square. a² + b² = {a * a} + {b * b} = {a * a + b * b}"
            )
        elif kind == "factoring":
            a = rng.randint(2, 6)
            b = rng.randint(1, 8)
            fmt = (
                f"Student factored {a}x² + {a * b}x as {a}x(x + {a * b}).\n"
                f"Find the error.\n\n"
                f"The error: when factoring out {a}x, divide each term by {a}x.\n"
                f"{a}x² / {a}x = x and {a * b}x / {a}x = {b}\n"
                f"Correct: {a}x(x + {b})"
            )
        elif kind == "distribution":
            a = rng.randint(2, 8)
            b, c = rng.randint(1, 10), rng.randint(1, 10)
            fmt = (
                f"Student expanded {a}(x + {b}) as {a}x + {b}.\n"
                f"Find the error.\n\n"
                f"The error: the student only multiplied the first term.\n"
                f"Correct distribution: {a} × x + {a} × {b} = {a}x + {a * b}"
            )
        elif kind == "division":
            a = rng.randint(2, 10)
            b = rng.randint(1, 10)
            c = rng.randint(1, 5)
            result = a * c + b
            fmt = (
                f"Student divided {result} by {a}:\n"
                f"{result} ÷ {a} = {c} remainder {b + a}\n\n"
                f"Find the error.\n\n"
                f"The remainder is wrong. {result} = {a} × {c} + {result - a * c}\n"
                f"Correct: {result} ÷ {a} = {c} remainder {result - a * c}"
            )
        elif kind == "fraction":
            num1, den1 = rng.randint(1, 5), rng.randint(2, 8)
            num2, den2 = rng.randint(1, 5), rng.randint(2, 8)
            wrong_num = num1 + num2
            wrong_den = den1 + den2
            fmt = (
                f"Student added {num1}/{den1} + {num2}/{den2}:\n"
                f"Result: {wrong_num}/{wrong_den}\n\n"
                f"Find the error.\n\n"
                f"The error: you cannot add numerators and denominators directly.\n"
                f"Correct: find common denominator {den1 * den2}\n"
                f"{num1}/{den1} = {num1 * den2}/{den1 * den2}\n"
                f"{num2}/{den2} = {num2 * den1}/{den1 * den2}\n"
                f"Sum = {num1 * den2 + num2 * den1}/{den1 * den2}"
            )
        templates.append(fmt)
    return templates


# ─── Expression Simplification ─────────────────────────────────────

def gen_simplification(rng, n):
    """Expression simplification with steps."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["expand", "combine", "rationalize", "exponent"])
        if kind == "expand":
            a = rng.randint(1, 10)
            b = rng.randint(1, 10)
            c = rng.randint(1, 10)
            fmt = (
                f"Simplify: {a}(x + {b}) + {c}x\n"
                f"= {a}x + {a * b} + {c}x\n"
                f"= {a + c}x + {a * b}"
            )
        elif kind == "combine":
            a = rng.randint(1, 10)
            b = rng.randint(1, 10)
            c = rng.randint(1, 5)
            d = rng.randint(1, 10)
            fmt = (
                f"Simplify: {a}x² + {b}x - {c}x² + {d}x\n"
                f"= ({a} - {c})x² + ({b} + {d})x\n"
                f"= {a - c}x² + {b + d}x"
            )
        elif kind == "rationalize":
            a = rng.randint(2, 10)
            fmt = (
                f"Rationalize: 1/√{a}\n"
                f"= 1/√{a} × √{a}/√{a}\n"
                f"= √{a}/{a}"
            )
        elif kind == "exponent":
            a = rng.randint(2, 5)
            b = rng.randint(2, 4)
            c = rng.randint(2, 3)
            fmt = (
                f"Simplify: ({a}^{b})^{c}\n"
                f"= {a}^({b} × {c})\n"
                f"= {a}^{b * c}\n"
                f"= {a ** (b * c)}"
            )
        templates.append(fmt)
    return templates


# ─── Code-as-Solver ────────────────────────────────────────────────

def gen_code_solver(rng, n):
    """Python code solutions to math problems."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["sum_range", "factorial", "gcd", "prime", "fibonacci",
                           "quadratic", "statistics", "combinatorics"])
        if kind == "sum_range":
            a = rng.randint(1, 50)
            b = rng.randint(a + 10, 100)
            ans = sum(range(a, b + 1))
            fmt = (
                f"Problem: What is the sum of all integers from {a} to {b}?\n"
                f"\nPython solution:\n"
                f"result = sum(range({a}, {b + 1}))\n"
                f"print(result)  # {ans}\n"
                f"\nAnswer: {ans}"
            )
        elif kind == "factorial":
            n_val = rng.randint(3, 10)
            ans = math.factorial(n_val)
            fmt = (
                f"Problem: What is {n_val}!?\n"
                f"\nPython solution:\n"
                f"import math\n"
                f"result = math.factorial({n_val})\n"
                f"print(result)  # {ans}\n"
                f"\nAnswer: {ans}"
            )
        elif kind == "gcd":
            a = rng.randint(10, 50)
            b = rng.randint(10, 50)
            from math import gcd
            ans = gcd(a, b)
            fmt = (
                f"Problem: What is the GCD of {a} and {b}?\n"
                f"\nPython solution:\n"
                f"import math\n"
                f"result = math.gcd({a}, {b})\n"
                f"print(result)  # {ans}\n"
                f"\nAnswer: {ans}"
            )
        elif kind == "prime":
            n_val = rng.choice([11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
            fmt = (
                f"Problem: Is {n_val} prime?\n"
                f"\nPython solution:\n"
                f"n = {n_val}\n"
                f"is_prime = all(n % i != 0 for i in range(2, int(n**0.5) + 1))\n"
                f"print(is_prime)  # True\n"
                f"\nAnswer: Yes, {n_val} is prime"
            )
        elif kind == "fibonacci":
            n_val = rng.randint(6, 15)
            fib = [0, 1]
            for _ in range(2, n_val + 1):
                fib.append(fib[-1] + fib[-2])
            fmt = (
                f"Problem: What is the {n_val}th Fibonacci number?\n"
                f"\nPython solution:\n"
                f"a, b = 0, 1\n"
                f"for _ in range({n_val}):\n"
                f"    a, b = b, a + b\n"
                f"print(a)  # {fib[n_val]}\n"
                f"\nAnswer: {fib[n_val]}"
            )
        elif kind == "quadratic":
            a = rng.randint(1, 5)
            r1 = rng.randint(-8, 8)
            r2 = rng.randint(-8, 8)
            b = -a * (r1 + r2)
            c = a * r1 * r2
            fmt = (
                f"Problem: Solve {a}x^2 + ({b})x + ({c}) = 0\n"
                f"\nPython solution:\n"
                f"import math\n"
                f"a_coeff, b_coeff, c_coeff = {a}, {b}, {c}\n"
                f"discriminant = b_coeff**2 - 4*a_coeff*c_coeff\n"
                f"x1 = (-b_coeff + math.sqrt(discriminant)) / (2*a_coeff)\n"
                f"x2 = (-b_coeff - math.sqrt(discriminant)) / (2*a_coeff)\n"
                f"print(f'x1={{x1}}, x2={{x2}}')  # x1={float(r1)}, x2={float(r2)}\n"
                f"\nAnswer: x = {r1} or x = {r2}"
            )
        elif kind == "statistics":
            nums = [rng.randint(1, 50) for _ in range(rng.randint(5, 10))]
            mean = sum(nums) / len(nums)
            var = sum((x - mean) ** 2 for x in nums) / len(nums)
            std = var ** 0.5
            fmt = (
                f"Problem: Find the mean and standard deviation of {nums}\n"
                f"\nPython solution:\n"
                f"data = {nums}\n"
                f"mean = sum(data) / len(data)\n"
                f"variance = sum((x - mean) ** 2 for x in data) / len(data)\n"
                f"std = variance ** 0.5\n"
                f"print(f'Mean: {{mean:.2f}}, Std: {{std:.2f}}')\n"
                f"\nAnswer: Mean = {mean:.2f}, Standard deviation = {std:.2f}"
            )
        elif kind == "combinatorics":
            n = rng.randint(5, 15)
            r = rng.randint(2, min(n - 1, 5))
            ans = math.comb(n, r)
            fmt = (
                f"Problem: How many ways to choose {r} from {n}?\n"
                f"\nPython solution:\n"
                f"import math\n"
                f"result = math.comb({n}, {r})\n"
                f"print(result)  # {ans}\n"
                f"\nAnswer: {ans}"
            )
        templates.append(fmt)
    return templates


# ─── Advanced Algebra ───────────────────────────────────────────────

def gen_advanced_algebra(rng, n):
    """Level 4: Polynomials, logarithms, exponentials, abs value, fractional eq, determinants."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["linear_system", "polynomial", "log", "exponential",
                           "abs_value", "fractional", "determinant", "quadratic_formula"])
        if kind == "linear_system":
            x = rng.randint(-5, 5)
            y = rng.randint(-5, 5)
            a1, b1 = rng.randint(1, 4), rng.randint(1, 4)
            a2, b2 = rng.randint(1, 4), rng.randint(1, 4)
            while a1 * b2 == a2 * b1:
                a2, b2 = rng.randint(1, 4), rng.randint(1, 4)
            c1 = a1 * x + b1 * y
            c2 = a2 * x + b2 * y
            fmt = (
                f"Solve the system:\n"
                f"{a1}x + {b1}y = {c1}\n"
                f"{a2}x + {b2}y = {c2}\n"
                f"Answer: x = {x}, y = {y}\n"
                f"Check: {a1}({x}) + {b1}({y}) = {c1} ✓ ; {a2}({x}) + {b2}({y}) = {c2} ✓"
            )
        elif kind == "polynomial":
            coeffs = [rng.randint(1, 6) for _ in range(3)]
            x = rng.randint(-3, 3)
            val = coeffs[2] * x**2 + coeffs[1] * x + coeffs[0]
            fmt = (
                f"Evaluate f(x) = {coeffs[2]}x² + {coeffs[1]}x + {coeffs[0]} at x = {x}.\n"
                f"f({x}) = {coeffs[2]}({x})² + {coeffs[1]}({x}) + {coeffs[0]}\n"
                f"= {coeffs[2]}({x**2}) + {coeffs[1] * x} + {coeffs[0]}\n"
                f"= {val}"
            )
        elif kind == "log":
            base = rng.choice([2, 3, 5, 10])
            exp = rng.randint(2, 4)
            num = base ** exp
            fmt = rng.choice([
                f"Evaluate log_{base}({num}).  Answer: {exp}",
                f"What is log_{base}({num})?  Hint: {base}^{exp} = {num}.  Answer: {exp}",
                f"log_{base}({num}) = ?  Answer: {exp}",
            ])
        elif kind == "exponential":
            base = rng.choice([2, 3, 4, 5])
            exp = rng.randint(1, 4)
            ans = base ** exp
            fmt = (
                f"Solve 2^{exp} = {ans}, and more generally {base}^{exp} = {base ** exp}.\n"
                f"Evaluate {base}^{exp} = {ans}"
            )
        elif kind == "abs_value":
            a = rng.randint(-12, 12)
            while a == 0:
                a = rng.randint(-12, 12)
            fmt = rng.choice([
                f"Evaluate |{a}|. Answer: {abs(a)}",
                f"Find the absolute value of {a}. Answer: {abs(a)}",
                f"|{a}| = ? Answer: {abs(a)}",
            ])
        elif kind == "fractional":
            a = rng.randint(1, 12)
            b = rng.randint(1, 8)
            x = rng.randint(1, 10)
            # (a/x) = b  => x = a/b when divisible
            if a % b == 0:
                x_val = a // b
                fmt = (
                    f"Solve {a}/x = {b}.\n"
                    f"x = {a}/{b} = {x_val}\n"
                    f"Answer: x = {x_val}"
                )
            else:
                x = rng.randint(1, 10)
                num = a * x
                fmt = (
                    f"Solve {num}/x = {a}.\n"
                    f"x = {num}/{a} = {x}\n"
                    f"Answer: x = {x}"
                )
        elif kind == "determinant":
            a, b = rng.randint(1, 6), rng.randint(1, 6)
            c, d = rng.randint(1, 6), rng.randint(1, 6)
            det = a * d - b * c
            fmt = (
                f"Find the determinant of the 2x2 matrix:\n"
                f"[ {a}  {b} ]\n"
                f"[ {c}  {d} ]\n"
                f"det = {a}×{d} - {b}×{c} = {a * d} - {b * c} = {det}\n"
                f"Answer: {det}"
            )
        elif kind == "quadratic_formula":
            r1 = rng.randint(-5, 5)
            r2 = rng.randint(-5, 5)
            # a=1: x^2 - (r1+r2)x + r1*r2 = 0
            b = -(r1 + r2)
            c = r1 * r2
            fmt = (
                f"Solve x² + ({b})x + ({c}) = 0 using factorization.\n"
                f"Find two numbers that multiply to {c} and add to {b}: {r1} and {r2}.\n"
                f"(x {r1:+d})(x {r2:+d}) = 0\n"
                f"Answer: x = {-r1} or x = {-r2}"
            )
        templates.append(fmt)
    return templates


# ─── Number Theory ──────────────────────────────────────────────────

def gen_number_theory(rng, n):
    """Level 4: Primes, divisibility, GCD/LCM, modular arithmetic, digits."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["prime_check", "gcd", "lcm", "mod", "divisibility", "digits", "factors"])
        if kind == "prime_check":
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
            comp = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
            is_prime = rng.random() < 0.5
            num = rng.choice(primes if is_prime else comp)
            fmt = rng.choice([
                f"Is {num} a prime number? Answer: {'Yes' if is_prime else 'No'}",
                f"Tell whether {num} is prime or composite. Answer: {'prime' if is_prime else 'composite'}",
            ])
        elif kind == "gcd":
            from math import gcd
            a, b = rng.randint(12, 60), rng.randint(12, 60)
            g = gcd(a, b)
            fmt = (
                f"Find the GCD of {a} and {b}.\n"
                f"Prime factorization or Euclidean algorithm gives gcd({a}, {b}) = {g}\n"
                f"Answer: {g}"
            )
        elif kind == "lcm":
            from math import gcd
            a, b = rng.randint(4, 20), rng.randint(4, 20)
            l = a * b // gcd(a, b)
            fmt = (
                f"Find the LCM of {a} and {b}.\n"
                f"LCM = {a} × {b} / gcd({a},{b}) = {a * b} / {gcd(a, b)} = {l}\n"
                f"Answer: {l}"
            )
        elif kind == "mod":
            a = rng.randint(10, 100)
            m = rng.choice([3, 4, 5, 7, 8, 9, 11, 12, 13])
            r = a % m
            fmt = rng.choice([
                f"Compute {a} mod {m}. Answer: {r}",
                f"What is {a} modulo {m}? Answer: {r}",
                f"The remainder when {a} is divided by {m} is? Answer: {r}",
            ])
        elif kind == "divisibility":
            num = rng.randint(100, 999)
            base = rng.choice([10, 100])
            # num divisible by something
            divisor = rng.choice([2, 3, 4, 5, 6, 8, 9, 10, 12])
            target = (num // divisor) * divisor
            fmt = (
                f"Is {target} divisible by {divisor}? Yes, because {target} = {divisor} × {target // divisor}.\n"
                f"Verification: {target} ÷ {divisor} = {target // divisor} (exact). Answer: Yes"
            )
        elif kind == "digits":
            num = rng.randint(2, 500)
            s = sum(int(d) for d in str(num))
            fmt = (
                f"Find the sum of the digits of {num}.\n"
                f"Digits: {', '.join(str(num))}\n"
                f"Sum = {s}\n"
                f"Answer: {s}"
            )
        elif kind == "factors":
            num = rng.choice([12, 18, 24, 30, 36, 40, 48, 60])
            facs = [i for i in range(1, num + 1) if num % i == 0]
            fmt = (
                f"List all factors of {num}.\n"
                f"Answer: {', '.join(map(str, facs))}"
            )
        templates.append(fmt)
    return templates


# ─── Geometry ───────────────────────────────────────────────────────

def gen_geometry(rng, n):
    """Level 4: Area, perimeter, volume, Pythagoras, angles."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["square", "rect", "triangle", "circle", "cube", "rect_prism", "pythagoras", "angles"])
        if kind == "square":
            s = rng.randint(3, 12)
            fmt = (
                f"A square has side length {s}. Find its area and perimeter.\n"
                f"Area = s² = {s}² = {s * s}\n"
                f"Perimeter = 4s = 4 × {s} = {4 * s}\n"
                f"Answer: Area = {s * s}, Perimeter = {4 * s}"
            )
        elif kind == "rect":
            l, w = rng.randint(4, 15), rng.randint(3, 10)
            fmt = (
                f"A rectangle has length {l} and width {w}. Find its area and perimeter.\n"
                f"Area = {l} × {w} = {l * w}\n"
                f"Perimeter = 2({l} + {w}) = {2 * (l + w)}\n"
                f"Answer: Area = {l * w}, Perimeter = {2 * (l + w)}"
            )
        elif kind == "triangle":
            b = rng.randint(4, 14)
            h = rng.randint(3, 12)
            area = b * h // 2
            fmt = (
                f"A triangle has base {b} and height {h}. Find its area.\n"
                f"Area = (1/2) × base × height = (1/2) × {b} × {h} = {area}\n"
                f"Answer: {area}"
            )
        elif kind == "circle":
            r = rng.choice([2, 3, 4, 5, 7, 10])
            area = 3.14 * r * r
            circ = 2 * 3.14 * r
            fmt = (
                f"A circle has radius {r} (use π ≈ 3.14). Find area and circumference.\n"
                f"Area = πr² = 3.14 × {r}² = {area:.2f}\n"
                f"Circumference = 2πr = 2 × 3.14 × {r} = {circ:.2f}\n"
                f"Answer: Area ≈ {area:.2f}, Circumference ≈ {circ:.2f}"
            )
        elif kind == "cube":
            s = rng.randint(2, 6)
            fmt = (
                f"A cube has side {s}. Find its volume and surface area.\n"
                f"Volume = s³ = {s}³ = {s ** 3}\n"
                f"Surface area = 6s² = 6 × {s ** 2} = {6 * s ** 2}\n"
                f"Answer: Volume = {s ** 3}, Surface area = {6 * s ** 2}"
            )
        elif kind == "rect_prism":
            l, w = rng.randint(2, 8), rng.randint(2, 8)
            h = rng.randint(2, 6)
            fmt = (
                f"A rectangular prism is {l} × {w} × {h}. Find its volume.\n"
                f"Volume = l × w × h = {l} × {w} × {h} = {l * w * h}\n"
                f"Answer: {l * w * h}"
            )
        elif kind == "pythagoras":
            p = rng.randint(3, 10)
            q = rng.randint(3, 10)
            hyp = p * p + q * q
            # ensure a perfect-square hypotenuse when possible
            import math as m
            r = math.isqrt(hyp)
            if r * r == hyp:
                fmt = (
                    f"A right triangle has legs {p} and {q}. Find the hypotenuse.\n"
                    f"c² = {p}² + {q}² = {p * p} + {q * q} = {hyp}\n"
                    f"c = √{hyp} = {r}\n"
                    f"Answer: {r}"
                )
            else:
                fmt = (
                    f"A right triangle has legs {p} and {q}. Find c² (hypotenuse squared).\n"
                    f"c² = {p}² + {q}² = {p * p} + {q * q} = {hyp}\n"
                    f"Answer: {hyp}"
                )
        elif kind == "angles":
            a = rng.randint(30, 150)
            comp = 90 - a if 90 - a >= 0 else None
            supp = 180 - a
            if comp is not None and rng.random() < 0.5:
                fmt = (
                    f"The complement of an angle of {a}° is?\n"
                    f"Complement = 90° - {a}° = {comp}°\n"
                    f"Answer: {comp}°"
                )
            else:
                fmt = (
                    f"The supplement of an angle of {a}° is?\n"
                    f"Supplement = 180° - {a}° = {supp}°\n"
                    f"Answer: {supp}°"
                )
        templates.append(fmt)
    return templates


# ─── Trigonometry ───────────────────────────────────────────────────

def gen_trig(rng, n):
    """Level 4: Standard angle values, basic identities."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["sin", "cos", "tan", "identity", "pythag_ident"])
        if kind == "sin":
            # standard angles
            ang = rng.choice([0, 30, 45, 60, 90])
            vals = {0: "0", 30: "1/2", 45: "√2/2", 60: "√3/2", 90: "1"}
            fmt = (
                f"What is sin({ang}°)?\n"
                f"From the unit circle, sin({ang}°) = {vals[ang]}\n"
                f"Answer: {vals[ang]}"
            )
        elif kind == "cos":
            ang = rng.choice([0, 30, 45, 60, 90])
            vals = {0: "1", 30: "√3/2", 45: "√2/2", 60: "1/2", 90: "0"}
            fmt = (
                f"What is cos({ang}°)?\n"
                f"From the unit circle, cos({ang}°) = {vals[ang]}\n"
                f"Answer: {vals[ang]}"
            )
        elif kind == "tan":
            ang = rng.choice([0, 30, 45, 60])
            vals = {0: "0", 30: "√3/3", 45: "1", 60: "√3"}
            fmt = (
                f"What is tan({ang}°)?\n"
                f"tan({ang}°) = sin/cos = {vals[ang]}\n"
                f"Answer: {vals[ang]}"
            )
        elif kind == "identity":
            a = rng.randint(2, 10)
            b = rng.randint(2, 10)
            # sin(A+B)
            fmt = (
                f"Write the expansion of sin(A + B), then evaluate if A = {a}° and B = {b}° conceptually.\n"
                f"sin(A + B) = sin A cos B + cos A sin B\n"
                f"Answer: sin A cos B + cos A sin B"
            )
        elif kind == "pythag_ident":
            val = rng.choice([1, 1, 1])
            fmt = (
                f"Simplify: sin²θ + cos²θ.\n"
                f"The Pythagorean identity states sin²θ + cos²θ = 1 for any angle θ.\n"
                f"Answer: 1"
            )
        templates.append(fmt)
    return templates


# ─── Combinatorics & Probability ────────────────────────────────────

def gen_combinatorics_prob(rng, n):
    """Level 3-4: Permutations, combinations, probability."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["permutation", "combination", "probability", "arrange", "fund_count"])
        if kind == "permutation":
            n = rng.randint(3, 8)
            r = rng.randint(1, n)
            p = math.factorial(n) // math.factorial(n - r)
            fmt = (
                f"How many ways to arrange {r} items from {n} (order matters)?\n"
                f"P({n}, {r}) = {n}! / ({n - r})! = {p}\n"
                f"Answer: {p}"
            )
        elif kind == "combination":
            n = rng.randint(5, 15)
            r = rng.randint(2, min(n - 1, 6))
            c = math.comb(n, r)
            fmt = (
                f"How many ways to choose {r} from {n} (order does not matter)?\n"
                f"C({n}, {r}) = {n}! / ({r}! × {n - r}!) = {c}\n"
                f"Answer: {c}"
            )
        elif kind == "probability":
            total = rng.randint(5, 20)
            favorable = rng.randint(1, total - 1)
            from math import gcd
            g = gcd(favorable, total)
            num, den = favorable // g, total // g
            fmt = (
                f"A bag has {total} balls: {favorable} red and {total - favorable} blue. "
                f"Draw one at random. P(red)?\n"
                f"P(red) = {favorable}/{total} = {num}/{den}\n"
                f"Answer: {num}/{den}"
            )
        elif kind == "arrange":
            n = rng.randint(3, 6)
            total = math.factorial(n)
            fmt = (
                f"How many ways to arrange {n} distinct objects?\n"
                f"{n}! = {total}\n"
                f"Answer: {total}"
            )
        elif kind == "fund_count":
            a = rng.randint(2, 8)
            b = rng.randint(2, 8)
            fmt = (
                f"You have {a} shirts and {b} pants. How many outfits?\n"
                f"By the fundamental counting principle: {a} × {b} = {a * b}\n"
                f"Answer: {a * b}"
            )
        templates.append(fmt)
    return templates


# ─── Verification ───────────────────────────────────────────────────

def gen_verification(rng, n):
    """Candidate solutions → verify correctness (both valid and invalid candidates)."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["valid_add", "invalid_add", "valid_eq", "invalid_eq",
                           "valid_sub", "invalid_sub"])
        if kind in ("valid_add", "invalid_add"):
            a, b = rng.randint(2, 50), rng.randint(2, 50)
            correct = a + b
            candidate = correct if kind == "valid_add" else correct + rng.randint(1, 4)
            verdict = "correct" if candidate == correct else "incorrect"
            fmt = (
                f"Problem: Calculate {a} + {b}.\n"
                f"Candidate solution: {candidate}\n"
                f"Verification: {a} + {b} = {correct}.\n"
                f"The candidate says {candidate}, so the candidate is {verdict}."
            )
        elif kind in ("valid_sub", "invalid_sub"):
            a = rng.randint(20, 80)
            b = rng.randint(2, a - 1)
            correct = a - b
            candidate = correct if kind == "valid_sub" else correct + rng.randint(1, 5)
            verdict = "correct" if candidate == correct else "incorrect"
            fmt = (
                f"Problem: Calculate {a} - {b}.\n"
                f"Candidate solution: {candidate}\n"
                f"Verification: {a} - {b} = {correct}.\n"
                f"The candidate says {candidate}, so the candidate is {verdict}."
            )
        elif kind in ("valid_eq", "invalid_eq"):
            a = rng.randint(2, 10)
            b = rng.randint(10, 60)
            x = rng.randint(2, 20)
            rhs = a * x + b
            candidate = x if kind == "valid_eq" else x + rng.choice([-2, -1, 1, 2])
            verdict = "correct" if candidate == x else "incorrect"
            fmt = (
                f"Problem: Solve {a}x + {b} = {rhs}.\n"
                f"Candidate solution: x = {candidate}\n"
                f"Verification: plug in x = {candidate}: {a}({candidate}) + {b} = {a * candidate + b}.\n"
                f"We need {rhs}. Since {a * candidate + b} ≠ {rhs}, the candidate is {verdict}."
                if a * candidate + b != rhs else
                f"Problem: Solve {a}x + {b} = {rhs}.\n"
                f"Candidate solution: x = {candidate}\n"
                f"Verification: plug in x = {candidate}: {a}({candidate}) + {b} = {a * candidate + b} = {rhs}.\n"
                f"The check matches, so the candidate is correct."
            )
        templates.append(fmt)
    return templates


# ─── Clean-Completion Math (natural-language Q&A) ──────────────────
# Full-sentence prompts and answers so the model learns to produce a
# clean, complete response instead of collapsing to SO/code garbage.

def gen_clean_completion(rng, n):
    """Level 1-2: Conversational Q&A with full-sentence responses.

    Each sample is a complete natural-language exchange:
      Q: What is 2 plus 2?
      A: 2 plus 2 equals 4.
    This teaches the model to *complete* cleanly rather than just emit
    a correct first token then collapse into StackOverflow formatting.
    """
    templates = []
    for _ in range(n):
        kind = rng.choice(["add", "sub", "mul", "div", "sqrt", "pow"])
        if kind == "add":
            a, b = rng.randint(0, 100), rng.randint(0, 100)
            ans = a + b
            fmt = rng.choice([
                f"Q: What is {a} plus {b}?\nA: {a} plus {b} equals {ans}.",
                f"Q: Add {a} and {b}.\nA: {a} added to {b} gives {ans}.",
                f"Q: {a} + {b} = ?\nA: {a} + {b} = {ans}.",
                f"Q: Can you tell me {a} + {b}?\nA: Certainly, {a} + {b} is {ans}.",
            ])
        elif kind == "sub":
            a = rng.randint(10, 200)
            b = rng.randint(0, a)
            ans = a - b
            fmt = rng.choice([
                f"Q: What is {a} minus {b}?\nA: {a} minus {b} equals {ans}.",
                f"Q: Subtract {b} from {a}.\nA: {a} - {b} = {ans}.",
                f"Q: {a} - {b} = ?\nA: {a} - {b} = {ans}.",
                f"Q: Take {b} away from {a}.\nA: That leaves {ans}.",
            ])
        elif kind == "mul":
            a, b = rng.randint(1, 20), rng.randint(1, 20)
            ans = a * b
            fmt = rng.choice([
                f"Q: What is {a} times {b}?\nA: {a} times {b} equals {ans}.",
                f"Q: Multiply {a} by {b}.\nA: {a} × {b} = {ans}.",
                f"Q: {a} × {b} = ?\nA: {a} × {b} = {ans}.",
                f"Q: How much is {a} multiplied by {b}?\nA: It is {ans}.",
            ])
        elif kind == "div":
            b = rng.randint(1, 15)
            q = rng.randint(1, 15)
            a = b * q
            fmt = rng.choice([
                f"Q: What is {a} divided by {b}?\nA: {a} divided by {b} equals {q}.",
                f"Q: Divide {a} by {b}.\nA: {a} / {b} = {q}.",
                f"Q: {a} / {b} = ?\nA: {a} / {b} = {q}.",
                f"Q: Share {a} equally among {b} people.\nA: Each person gets {q}.",
            ])
        elif kind == "sqrt":
            r = rng.randint(1, 15)
            sq = r * r
            fmt = rng.choice([
                f"Q: What is the square root of {sq}?\nA: The square root of {sq} is {r}.",
                f"Q: sqrt({sq}) = ?\nA: sqrt({sq}) = {r}.",
                f"Q: Find the square root of {sq}.\nA: It is {r}.",
            ])
        elif kind == "pow":
            base = rng.randint(2, 8)
            exp = rng.randint(2, 4)
            ans = base ** exp
            fmt = rng.choice([
                f"Q: What is {base} to the power {exp}?\nA: {base}^{exp} = {ans}.",
                f"Q: Compute {base}^{exp}.\nA: {base}^{exp} equals {ans}.",
                f"Q: {base} raised to {exp} is?\nA: {base} raised to {exp} is {ans}.",
            ])
        templates.append(fmt)
    return templates


# ─── Generator registry ─────────────────────────────────────────────
# (label, generator, approx tokens per doc)
GENERATORS = [
    ("arithmetic",        gen_arithmetic_direct,  55),
    ("cot_scratchpad",    gen_cot_scratchpad,    115),
    ("arithmetic_worked", gen_arithmetic_worked, 105),
    ("problem",           gen_problem_solving,   110),
    ("algebra",           gen_algebra_direct,     70),
    ("advanced_algebra",  gen_advanced_algebra,  110),
    ("number_theory",     gen_number_theory,      70),
    ("geometry",          gen_geometry,           90),
    ("trig",              gen_trig,               60),
    ("combinatorics",     gen_combinatorics_prob, 80),
    ("verification",      gen_verification,       60),
    ("error",             gen_error_identification, 100),
    ("simplify",          gen_simplification,     50),
    ("code",              gen_code_solver,        90),
    ("clean",             gen_clean_completion,   40),
]

# Relative weight of each generator (guide for --all allocation).
# These are not hard curriculum allocations (the curriculum builder does
# that); they just control how many tokens each generator produces when
# running --all. Advanced + verification are weighted up per the design.
GEN_WEIGHTS = {
    "arithmetic":        0.22,
    "cot_scratchpad":    0.14,
    "arithmetic_worked": 0.10,
    "problem":           0.08,
    "algebra":           0.13,
    "advanced_algebra":  0.10,
    "number_theory":     0.08,
    "geometry":          0.08,
    "trig":              0.06,
    "combinatorics":     0.08,
    "verification":      0.10,
    "error":             0.06,
    "simplify":          0.03,
    "code":              0.03,
    "clean":             0.03,
}


def write_docs(templates, tok, out_path, target_tokens):
    """Tokenize templates and write to bin file."""
    n = 0
    with open(out_path, "wb") as f:
        for text in templates:
            ids = tok.encode(text.strip())
            if not ids:
                continue
            if n + len(ids) > target_tokens:
                ids = ids[:target_tokens - n]
            if not ids:
                break
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += len(ids)
            if n >= target_tokens:
                break
    return n


def validate_generators(rng, per_gen=200):
    """Quality-check every generator: correctness, format, duplicates.

    Returns (report_lines, ok). Reports issues and prints a few samples.
    """
    report = []
    ok = True
    for label, gen, approx in GENERATORS:
        docs = gen(rng, per_gen)
        # checks
        empty = sum(1 for d in docs if not d or not d.strip())
        dup = len(docs) - len(set(docs))
        short = sum(1 for d in docs if len(d) < 10)
        # "no-answer" is informational only — different families use varied
        # phrasing (Answer/Correct/error/x=...); hard-fail is reserved for
        # structural breakage (empty docs, crashes).
        noans = sum(1 for d in docs
                    if not any(k in d for k in ("Answer", "=", "Solution",
                                                "Verification", "Correct:",
                                                "error", "A:")))
        report.append(
            f"{label:18s} n={len(docs):4d} empty={empty:3d} dup={dup:3d} "
            f"short={short:3d} no-ans={noans:3d}"
        )
        if empty:
            ok = False
        # print 2 samples for eyeballing
        for s in docs[:2]:
            report.append(f"    EX>> {s[:90]!r}")
    report.append("ALL GENERATORS OK" if ok else "ISSUES DETECTED - review above")
    return report, ok


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic math data")
    parser.add_argument("--all", action="store_true",
                        help="Generate all generators (default when --only not given)")
    parser.add_argument("--tokens", type=int, default=100_000_000,
                        help="Total synthetic tokens to generate (--all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Only generate these generators (e.g. --only arithmetic verification)")
    parser.add_argument("--validate", type=int, default=0,
                        help="Validate each generator with N sample docs, then exit")
    parser.add_argument("--per-gen-tokens", type=int, default=None,
                        help="Override per-generator token budget (for --only)")
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke test: scale targets to N total tokens")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    global EOS
    EOS = tok.eos_token
    rng = random.Random(args.seed)

    # Validation mode
    if args.validate > 0:
        print(f"Validating all {len(GENERATORS)} generators with {args.validate} samples each...")
        report, ok = validate_generators(rng, args.validate)
        print("\n".join(report))
        sys.exit(0 if ok else 1)

    # Choose generators to run
    if args.only:
        selected = [(l, g, a) for l, g, a in GENERATORS if l in args.only]
        if not selected:
            print(f"No generators matched {args.only}. Available: "
                  f"{[l for l,_,_ in GENERATORS]}")
            sys.exit(1)
    else:
        selected = GENERATORS

    total = args.smoke if args.smoke > 0 else args.tokens

    # Per-generator token budgets (weighted, or per-gen override)
    if args.per_gen_tokens:
        budgets = {l: args.per_gen_tokens for l, _, _ in selected}
    else:
        wsum = sum(GEN_WEIGHTS[l] for l, _, _ in selected)
        budgets = {l: int(total * GEN_WEIGHTS[l] / wsum) for l, _, _ in selected}

    print(f"Generating {total:,} synthetic math tokens across "
          f"{len(selected)} generators...\n")
    for label, _, _ in selected:
        print(f"  {label:20s} budget={budgets[label]:>12,} tokens")

    total_tokens = 0
    files = []
    for label, gen, approx in selected:
        n_docs = max(1, budgets[label] // approx)
        docs = gen(rng, n_docs)
        out_path = f"{DATA_DIR}/{label}_syn.bin"
        n = write_docs(docs, tok, out_path, budgets[label])
        print(f"  -> {out_path}  ({n:,}/{budgets[label]:,} tokens, {len(docs)} docs)")
        total_tokens += n
        files.append(out_path)

    print(f"\nTotal synthetic: {total_tokens:,} tokens")
    print("Files written to data/phase5/")


if __name__ == "__main__":
    main()
