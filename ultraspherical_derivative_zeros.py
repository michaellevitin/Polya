"""
ultraspherical_derivative_zeros.py

Certified rational enclosures for the zeros  p'_{d,m,k}  of the derivative of
the d-dimensional ultraspherical Bessel function of order m,

     P_{d,m}(x) := x^(-(d/2-1)) J_{m+d/2-1}(x),        d >= 3, m >= 0,

with the exceptional convention  p'_{d,0,1} := 0.

Python port of UltrasphericalDerivativeZeros.wl; both implement the algorithm of

  "Polya's conjecture for higher-dimensional Neumann balls",
  N. Filonov, M. Levitin, I. Polterovich, D. A. Sher,

specifically the procedures of the Appendix "Rational enclosure for zeros of
derivatives of ultraspherical Bessel functions", used to fill the finite gap in
Section 5 ("Filling the gap").  All numbering below refers to that paper:
Definition (defn:ratn), Lemmas (lem:enclosures), (lem:trans), (lem:pandj), the
Lorch-Szego lower bound (eq:LS), the sign conditions (eq:signconditions), and
the Polya inequality (eq:checkpolya).

NOTATION (as in the paper).
     nu   = m + d/2 - 1
     F_nu(t)     = Gamma(nu+1) (x/2)^(-nu) J_nu(x) |_{t = x^2/4}
                 = Sum_j (-t)^j / (j! Pi_j(nu))                    (eq:deff)
     H_{d,m}(t)  = m F_{m+d/2-1}(t) - (2t/(m+d/2)) F_{m+d/2}(t)    (eq:defh)
     P'_{d,m}(x) = x^(-d/2) (x/2)^(m+d/2-1) / Gamma(m+d/2)
                     * H_{d,m}(x^2/4)                              (eq:uprime)
Both F_nu and H_{d,m} are entire with RATIONAL Taylor coefficients (nu is a
half-integer), and H_{d,m}(0) = m.  In the notation of (defn:ratn),
f_{nu,k} = j_{nu,k}^2/4 are the positive zeros of F_nu, and
h_{d,m,k} = (p'_{d,m,k})^2/4 those of H_{d,m}.

CORRESPONDENCE WITH THE WOLFRAM PACKAGE.  The names match up to Python casing:

     NuOf / EtaOf / KapOf        <->  nu_of / eta_of / kap_of
     KappaMult                   <->  kappa_mult
     RatSqrtLower                <->  rat_sqrt_lower
     FEnclosure / HEnclosure     <->  f_enclosure / h_enclosure
     NStart                      <->  n_start
     SignF / SignH               <->  sign_f / sign_h
     BesselZeroBrackets          <->  bessel_zero_brackets
     RefineBracket               <->  refine_bracket
     ClassifyBracket             <->  classify_bracket
     DerivativeWindows           <->  derivative_windows
     MCutoff                     <->  m_cutoff
     ComputeListP                <->  compute_list_p
     NeumannCountingFunction     <->  neumann_counting_function
     WeylConstantBound           <->  weyl_constant_bound
     ExpandListP                 <->  expand_list_p
     PolyaRatio                  <->  polya_ratio
     GapTable / GapTableForm     <->  gap_table / gap_table_form
     PolyaCheckPassed            <->  polya_check_passed
     $LambdaStar                 <->  LAMBDA_STAR
     $GridStep / $MaxIter        <->  GRID_STEP / MAX_ITER

DESIGN RULE.  Every certificate rests exclusively on EXACT arithmetic, as
required in Section 5.  Here that means `fractions.Fraction` and Python's
unbounded integers throughout; no float and no float comparison occurs anywhere
in Sections 0-6 below.  Floating point appears ONLY in the clearly marked
diagnostic Section 7, which feeds into no certificate.

  *** Do not "optimise" any Fraction into a float. ***  Python will silently
  coerce Fraction to float in a mixed expression, so a stray float constant
  would quietly destroy the certification while still producing plausible
  numbers.  All literals below are int or Fraction for this reason.

VALIDATION.  Unlike the Wolfram file, this one was executed where it was
written.  Running it as a script reproduces the rows of Table (table:data);
d = 3 gives K_3 = 50, N^Neu = 570 and Polya ratio 0.90609..., and d = 4 gives
63, 4701 and 0.85332..., matching the paper.
"""

from fractions import Fraction
from math import comb, factorial, isqrt

# ============================================================================
# Section 0.  Elementary exact helpers
# ============================================================================

# Grid step of the march in bessel_zero_brackets.  The paper fixes the value 3,
# admissible because consecutive zeros satisfy j_{nu,k} - j_{nu,k-1} >= Pi > 3
# for nu >= 1/2 (Lorch-Szego, as quoted in the Certification clause of that
# procedure).
GRID_STEP = Fraction(3)

# Safety cap on the loops.  The paper PROVES termination of every loop
# (Certification/Termination clauses of the four procedures); the cap exists
# only to turn a hypothetical implementation bug into a clean exception rather
# than a hang.
MAX_ITER = 100000


class ImplementationBug(RuntimeError):
    """Raised when a safety cap is reached, which contradicts the proved
    termination and therefore indicates an implementation bug."""


def nu_of(d, m):
    """nu = m + d/2 - 1, exactly (Fraction, since d/2 may be a half-integer)."""
    return Fraction(m) + Fraction(d, 2) - 1


def eta_of(d):
    """eta = d/2 - 1, exactly."""
    return Fraction(d, 2) - 1


def kap_of(d, m):
    """The quantity m(m + d - 2) appearing in the Lorch-Szego bound (eq:LS)."""
    return m * (m + d - 2)


def kappa_mult(d, m):
    """Multiplicity kappa_{d,m} of the spherical harmonics H_{d,m} on S^(d-1).

    The single binomial formula is valid for all m >= 0 (comb(a, b) = 0 for
    integers 0 <= a < b), e.g. d = 3 gives 2m + 1.  Used to assemble the
    indices n_i and the counting function of Section 5.
    """
    hi = comb(m + d - 1, d - 1)
    lo = comb(m + d - 3, d - 1) if m + d - 3 >= 0 else 0
    return hi - lo


def rat_sqrt_lower(q, prec=16):
    """A certified RATIONAL lower bound for sqrt(q), q a Fraction >= 0, with
    dyadic denominator 2^prec.

    Uses integer square roots only, so the result x0 satisfies x0 <= sqrt(q)
    exactly.  Used to enter the zero-free initial interval in
    bessel_zero_brackets below.
    """
    if q < 0:
        raise ValueError("rat_sqrt_lower: negative argument")
    q = Fraction(q)
    # floor(sqrt(q) * 2^prec) = isqrt(floor(q * 2^(2 prec)))
    scaled = (q.numerator * (1 << (2 * prec))) // q.denominator
    return Fraction(isqrt(scaled), 1 << prec)


# ============================================================================
# Section 1.  Certified enclosures of F_nu(t) and H_{d,m}(t)
#             (Lemma lem:enclosures)
# ============================================================================

def f_enclosure(nu, t, n):
    """Rational interval (lo, hi) with lo <= F_nu(t) <= hi, or None if the
    hypothesis r_{nu,N}(t) < 1 of (lem:enclosures) is not yet met (in which
    case the caller simply increases n).

    The partial sum F_{nu,N}(t) is accumulated incrementally via the exact term
    recursion  a_0 = 1,  a_{j+1} = a_j (-t)/((j+1)(nu+j+1)),  after which
        tau = |a_{N+1}| = t^(N+1) / ((N+1)! Pi_{N+1}(nu)),
        rho = r_{nu,N}(t) = t / ((N+2)(nu+N+2)),
    and the radius is  delta_{F_{nu,N}}(t) = tau/(1 - rho),  exactly as in
    (lem:enclosures).  Every quantity is a Fraction.
    """
    term = Fraction(1)
    s = Fraction(1)
    for j in range(n):
        term = term * (-t) / ((j + 1) * (nu + j + 1))
        s += term
    # now s = F_{nu,n}, term = a_n
    tau = abs(term) * t / ((n + 1) * (nu + n + 1))      # = |a_{n+1}|
    rho = t / ((n + 2) * (nu + n + 2))                  # = r_{nu,n}(t)
    if rho >= 1:
        return None
    r = tau / (1 - rho)                                 # = delta_{F_{nu,n}}
    return (s - r, s + r)


def h_enclosure(d, m, t, n):
    """Rational interval containing H_{d,m}(t), obtained from the two
    F-enclosures by exact interval arithmetic.

    Since t >= 0 and the coefficients m and 2t/(nu+1) = 2t/(m + d/2) are >= 0,
    midpoints and radii combine linearly, giving exactly the H_{d,m,N} and
    delta_{H_{d,m,N}} of (lem:enclosures).
    """
    nu = nu_of(d, m)
    e1 = f_enclosure(nu, t, n)
    e2 = f_enclosure(nu + 1, t, n)
    if e1 is None or e2 is None:
        return None
    c = 2 * t / (nu + 1)                                # = 2t/(m + d/2)
    mid = m * (e1[0] + e1[1]) / 2 - c * (e2[0] + e2[1]) / 2
    rad = m * (e1[1] - e1[0]) / 2 + c * (e2[1] - e2[0]) / 2
    return (mid - rad, mid + rad)


# ============================================================================
# Section 2.  Procedure Sign(X, x)
# ============================================================================

def n_start(t):
    """Starting truncation order.

    The paper's workflow sets N = 8; any starting value is admissible, and
    n ~ 2 sqrt(t) merely starts past the hump of the series so that the radius
    already decays fast.  Computed with integer square roots, so this stays
    within exact arithmetic.
    """
    if t <= 0:
        return 8
    # smallest n with n^2 >= 4t, i.e. ceil(2 sqrt(t))
    n = isqrt((4 * t).numerator // (4 * t).denominator)
    while Fraction(n * n) < 4 * t:
        n += 1
    return max(8, n)


def _certified_sign(enclosure_fn, t, tag):
    """Shared doubling loop of the procedure Sign(X, x).

    Compute the enclosure; if it excludes 0, return the common sign of its
    endpoints; otherwise double N, as in the workflow.
    Certification: by (lem:enclosures).
    Termination: by (lem:trans), all zeros of F_nu and H_{d,m} are irrational,
    so the true value X(x) is nonzero at rational x and the loop must exit.
    """
    n = n_start(t)
    for _ in range(MAX_ITER):
        enc = enclosure_fn(t, n)
        if enc is not None and (enc[0] > 0 or enc[1] < 0):
            return 1 if enc[0] > 0 else -1
        n *= 2
    raise ImplementationBug(
        f"Internal safety cap reached in {tag}; this contradicts the proved "
        "termination and indicates an implementation bug."
    )


def sign_f(nu, x):
    """Certified sign of F_nu(x^2/4) in {+1, -1}, for rational x >= 0.

    At x = 0 the exact value F_nu(0) = 1 is used instead of an enclosure.
    """
    if x == 0:
        return 1                                        # F_nu(0) = 1
    t = x * x / 4
    return _certified_sign(lambda tt, nn: f_enclosure(nu, tt, nn), t, "sign_f")


def sign_h(d, m, x):
    """Certified sign of H_{d,m}(x^2/4) in {+1, -1}, for rational x >= 0.

    At x = 0 the exact value H_{d,m}(0) = m is used instead of an enclosure;
    this is the exception noted in the Input clause of the procedure Refine.
    """
    if x == 0:
        return (m > 0) - (m < 0)                        # H_{d,m}(0) = m
    t = x * x / 4
    return _certified_sign(lambda tt, nn: h_enclosure(d, m, tt, nn), t, "sign_h")


# ============================================================================
# Section 3.  Procedure BesselZeroBrackets(nu, Lambda)
# ============================================================================

def bessel_zero_brackets(nu, Lam):
    """Return the list of brackets

        [ [a_1, b_1, sa_1, sb_1], ..., [a_K, b_K, sa_K, sb_K] ]

    with j_{nu,k} in (a_k, b_k), b_k - a_k = 3, a_K >= Lambda, together with the
    certified signs sa_k, sb_k of F_nu at the endpoints (cached for the later
    bisections).

    Certification (as in the paper): consecutive zeros satisfy
    j_{nu,k} - j_{nu,k-1} >= Pi > 3 for nu >= 1/2 (Lorch-Szego), so each
    interval of length 3 contains at most one zero, which is simple; the march
    starts inside the zero-free initial interval, since j_{nu,1} > nu.  Rational
    grid points are never zeros by (lem:trans), so all signs are genuine.  Hence
    the recorded brackets enumerate, in increasing order and without omission,
    the zeros j_{nu,1}, j_{nu,2}, ... .
    Termination: there are finitely many zeros in (0, Lambda].

    NOTE ON THE ENTRY POINT.  The paper's workflow starts at x_0 = nu.  The code
    starts instead at a rational lower bound for sqrt(nu^2 - 1/4) <= nu, which
    lies in the same zero-free interval and is therefore equally valid, merely
    taking a few more grid steps.  To match the paper's workflow exactly,
    replace the assignment to x0 by  x0 = nu  (nu is a half-integer, hence
    exactly rational, so this stays within exact arithmetic).
    """
    x0 = Fraction(0) if nu == Fraction(1, 2) else rat_sqrt_lower(nu * nu - Fraction(1, 4))
    xi, si = x0, sign_f(nu, x0)         # si = +1, as j_{nu,1} > nu >= x0
    brackets = []
    for _ in range(MAX_ITER):
        xn = xi + GRID_STEP
        sn = sign_f(nu, xn)
        if si * sn < 0:                 # certified sign change:
            brackets.append([xi, xn, si, sn])       # exactly one zero here
            if xi >= Lam:                           # stop: a_K >= Lambda
                return brackets
        xi, si = xn, sn
    raise ImplementationBug("Internal safety cap reached in bessel_zero_brackets.")


# ============================================================================
# Section 4.  Procedure Refine(X, (a,b))   [code name: refine_bracket]
# ============================================================================

def refine_bracket(sign_fun, bracket):
    """One bisection step on a bracket [a, b, sa, sb] carrying a certified sign
    change of the function whose sign oracle is sign_fun (a callable
    x |-> +-1); at a zero left endpoint the exact value H_{d,m}(0) = m is used
    in place of Sign(X, 0), as stated in the Input clause of the procedure.

    The rational midpoint is never a zero by (lem:trans), so the returned
    half-width bracket again carries a certified sign change and contains the
    same unique zero.
    """
    l, r, sl, sr = bracket
    mid = (l + r) / 2                   # dyadic-friendly exact midpoint
    sm = sign_fun(mid)
    return [mid, r, sm, sr] if sm == sl else [l, mid, sl, sm]


# ============================================================================
# Section 5.  Procedure Main(d, Lambda, eps)
# ============================================================================

def classify_bracket(sign_fun, bracket, Lam, eps):
    """The two exit conditions (i) / (ii) shared by step 1 and step 2.3 of Main.

    Given a bracket certified to contain a unique zero z of the function with
    oracle sign_fun, bisect until either
      (i)  b <= Lambda and b - a <= eps: return ("include", a, b), the final
           two-sided enclosure a < z < b <= min(Lambda, z + eps); or
      (ii) a >= Lambda: return ("exclude",).
    Exactly one exit is reached after finitely many steps: by (lem:trans),
    z != Lambda, so z < Lambda forces (i) and z > Lambda forces (ii).
    """
    br = list(bracket)
    for _ in range(MAX_ITER):
        if br[1] <= Lam and br[1] - br[0] <= eps:
            return ("include", br[0], br[1])
        if br[0] >= Lam:
            return ("exclude",)
        br = refine_bracket(sign_fun, br)
    raise ImplementationBug("Internal safety cap reached in classify_bracket.")


def derivative_windows(d, m, dir_brs):
    """Steps 2.1 and 2.2 of Main, for m >= 1.

    Step 2.1: refine each bracket (a_k, b_k) of bessel_zero_brackets, bisecting
    on F_nu, until the ultraspherical derivative sign conditions

        Sign(H_{d,m}, a_k) = Sign(H_{d,m}, b_k) = (-1)^k     (eq:signconditions)

    hold.  Termination: as a_k increases to j_{nu,k} and b_k decreases to
    j_{nu,k}, both signs eventually equal (-1)^k by (lem:pandj), which gives
    sign P'_{d,m}(j_{nu,k}) = (-1)^k; each refine_bracket halves the width.
    Refining preserves a_K >= Lambda, since a_K can only increase.

    Step 2.2: return the ultraspherical derivative windows

        W_0 = (0, a_1),      W_k = (b_k, a_{k+1}),   k = 1, ..., K-1,

    each as a bracket-with-signs for the H_{d,m} oracle; at the left endpoint of
    W_0 the exact value H_{d,m}(0) = m > 0 is used.  By (lem:pandj) and
    (eq:signconditions), W_k contains exactly one zero of H_{d,m}, namely
    p'_{d,m,k+1}; no zero of H_{d,m} in (0, j_{nu,K}) lies outside the closure of
    the union of the W_k, while p'_{d,m,K+1} > j_{nu,K} > a_K >= Lambda.

    Subtle case handled automatically: consecutive zeros j_{nu,k} may fall in
    ADJACENT grid cells (their gap can be as small as Pi > 3), so that initially
    b_k = a_{k+1}.  The sign conditions at that shared point would be
    contradictory ((-1)^k versus (-1)^(k+1)), so the conditioning loop
    necessarily keeps refining until b_k and a_{k+1} separate; the certified
    signs then force b_k < p'_{d,m,k+1} < a_{k+1}, a nonempty window.
    """
    nu = nu_of(d, m)
    K = len(dir_brs)
    brs = [list(b) for b in dir_brs]
    # -- Step 2.1: enforce the sign conditions (eq:signconditions) ----------
    for k in range(1, K + 1):
        target = (-1) ** k
        for _ in range(MAX_ITER):
            if (sign_h(d, m, brs[k - 1][0]) == target
                    and sign_h(d, m, brs[k - 1][1]) == target):
                break
            brs[k - 1] = refine_bracket(lambda x: sign_f(nu, x), brs[k - 1])
        else:
            raise ImplementationBug(
                "Internal safety cap reached in derivative_windows.")
    # -- Step 2.2: assemble the windows W_0, ..., W_{K-1} -------------------
    windows = []
    for k in range(0, K):
        if k == 0:
            windows.append([Fraction(0), brs[0][0], 1, -1])
        else:
            windows.append([brs[k - 1][1], brs[k][0], (-1) ** k, (-1) ** (k + 1)])
    return windows


def m_cutoff(d, Lam):
    """The cut-off M_{d,Lambda} = max{ m >= 0 : m(m + d - 2) < Lambda^2 } quoted
    in Main, justified by the Lorch-Szego lower bound (eq:LS),
    p'_{d,m,1} > sqrt(m(m + d - 2)) for m >= 1.  Exact integer march.
    """
    M = 0
    while (M + 1) * (M + d - 1) < Lam * Lam:
        M += 1
    return M


def compute_list_p(d, Lam, eps):
    """The procedure Main(d, Lambda, eps).

    Returns the complete list ListP_{d,Lambda} of all pairs (m, k) with
    p'_{d,m,k} <= Lambda, as dicts

        {"m": m, "k": k, "lower": ..., "upper": ...}

    where "upper" is the certified rational \\overline{p'_{d,m,k}} with
        p'_{d,m,k} < upper <= min(p'_{d,m,k} + eps, Lambda),
    and "lower" the companion \\underline{p'_{d,m,k}} < p'_{d,m,k}, so that the
    two-sided enclosure of width at most eps mentioned in Section 5 is
    available.  For the exceptional zero the entry (0, 1, 0) is returned.
    """
    if not isinstance(d, int) or d < 3:
        raise ValueError("compute_list_p: d must be an integer >= 3")
    Lam, eps = Fraction(Lam), Fraction(eps)
    if Lam <= 0 or eps <= 0:
        raise ValueError("compute_list_p: Lambda and eps must be positive")

    L = []
    # ---- Step 0: initialise with the exceptional zero p'_{d,0,1} = 0 ------
    L.append({"m": 0, "k": 1, "lower": Fraction(0), "upper": Fraction(0)})

    # ---- Step 1: the case m = 0 ------------------------------------------
    # By (lem:pandj), p'_{d,0,k} = j_{d/2,k-1}, so the zeros are classified
    # directly on the brackets for nu = d/2 using the F-oracle.  They increase
    # in k, so we may stop at the first exclusion; by construction a_K >= Lam,
    # so exit (ii) occurs at the latest at k = K, and all zeros with k > K
    # satisfy j_{d/2,k} > j_{d/2,K} > a_K >= Lambda.
    nu = Fraction(d, 2)
    brs = bessel_zero_brackets(nu, Lam)
    for k, br in enumerate(brs, start=1):
        res = classify_bracket(lambda x: sign_f(nu, x), br, Lam, eps)
        if res[0] == "include":
            L.append({"m": 0, "k": k + 1, "lower": res[1], "upper": res[2]})
        else:
            break

    # ---- Step 2: the cases 1 <= m <= M_{d,Lambda} -------------------------
    M = m_cutoff(d, Lam)
    for m in range(1, M + 1):
        nu = nu_of(d, m)
        brs = bessel_zero_brackets(nu, Lam)      # exhaustive j_{nu,k}, k <= K
        wins = derivative_windows(d, m, brs)     # W_0, ..., W_{K-1}
        for k, w in enumerate(wins, start=1):
            # window W_{k-1} contains exactly one zero of H_{d,m}: p'_{d,m,k}
            res = classify_bracket(lambda x, mm=m: sign_h(d, mm, x), w, Lam, eps)
            if res[0] == "include":
                L.append({"m": m, "k": k, "lower": res[1], "upper": res[2]})
            else:
                break                            # zeros increase in k

    # ---- Step 3: return ListP --------------------------------------------
    return sorted(L, key=lambda e: (e["m"], e["k"]))


def neumann_counting_function(list_p, d):
    """Exact value of the Neumann counting function of the unit ball at Lambda,
    the fourth column of Table (table:data):
        N^Neu_{B^d}(Lambda) = sum over (m,k) in ListP of kappa_{d,m}.
    """
    return sum(kappa_mult(d, e["m"]) for e in list_p)


# ============================================================================
# Section 6.  Assembling the table of Section 5 and checking Polya's
#             conjecture, i.e. the inequality (eq:checkpolya)
# ============================================================================

def weyl_constant_bound(d):
    """A rational UPPER bound for the Weyl constant

        w_d = 2^(-d) / Gamma(1 + d/2)^2

    (the constant of Weyl's law for the unit ball).  For even d this is already
    rational: Gamma(1 + d/2) = (d/2)!.  For odd d = 2n+1 the duplication formula
    gives Gamma(n + 3/2) = (2n+2)! / (4^(n+1) (n+1)!) * sqrt(pi), so a factor
    1/pi remains, and -- exactly as stated in Section 5 -- we replace 1/pi by
    1/3.  Since pi > 3 this OVERSTATES w_d, so the resulting ratio overstates
    the left-hand side of (eq:checkpolya) and the check stays rigorous.

    The result is asserted to be a Fraction before it is returned, so a silent
    failure to clear the transcendental factor cannot slip through into a
    "certified" number.
    """
    if d % 2 == 0:
        g = factorial(d // 2)                       # Gamma(1 + d/2)
        w = Fraction(1, 2 ** d) / Fraction(g) ** 2
    else:
        n = (d - 1) // 2                            # d = 2n + 1
        # Gamma(1 + d/2) = Gamma(n + 3/2) = c * sqrt(pi),  c rational:
        c = Fraction(factorial(2 * n + 2), 4 ** (n + 1) * factorial(n + 1))
        # w_d = 2^-d / (c^2 pi);  replace 1/pi by 1/3:
        w = Fraction(1, 2 ** d) / (c ** 2) / 3
    if not isinstance(w, Fraction):
        raise TypeError(f"weyl_constant_bound({d}) did not reduce to a rational")
    return w


def expand_list_p(d, list_p, digits=6):
    """Re-arrange and re-index ListP exactly as in Section 5: sort the entries by
    their certified upper bounds and give them the cumulative indices

        n_1 = 1,     n_{i+1} = n_i + kappa_{d,m_i},

    so that the Neumann eigenvalues of B^d satisfy

        mu_{n_i} = ... = mu_{n_i + kappa_{d,m_i} - 1} < (upper_i)^2.

    The exceptional entry (0,1,0) has upper bound 0 and therefore sorts first,
    giving (n_1, m_1, k_1) = (1, 0, 1) as required.

    Why sorting by the UPPER bounds is legitimate even though the enclosures may
    in principle overlap: every entry placed before position i has upper bound
    <= upper_i, hence its true zero is < upper_i.  So the n_i - 1 eigenvalues
    accumulated before position i are all < (upper_i)^2 whatever the true order
    of two nearly equal zeros happens to be, and the displayed inequality for
    mu_{n_i} holds regardless.  Ties are broken by (m, k) purely for
    reproducibility.

    The exact rationals are retained in "lower"/"upper"; "lower(N)"/"upper(N)"
    are decimal approximations for display only and enter no certificate.
    """
    s = sorted(list_p, key=lambda e: (e["upper"], e["m"], e["k"]))
    out = []
    n = 1
    for e in s:
        kap = kappa_mult(d, e["m"])
        out.append({
            "n": n, "m": e["m"], "k": e["k"], "multiplicity": kap,
            "lower": e["lower"], "upper": e["upper"],
            "lower(N)": round(float(e["lower"]), digits),
            "upper(N)": round(float(e["upper"]), digits),
        })
        n += kap
    return out


def polya_ratio(d, expanded):
    """The left-hand side of (eq:checkpolya),

        max_{i >= 2}  w_d (upper_i)^d / (n_i - 1),

    as an EXACT Fraction (all inputs are rational, w_d being the bound above).
    The first entry, with n_1 = 1, is excluded: it would divide by n_1 - 1 = 0.
    Polya's conjecture holds on the range covered by ListP precisely when this
    value is < 1.
    """
    w = weyl_constant_bound(d)
    sel = [e for e in expanded if e["n"] > 1]
    if not sel:
        raise ValueError(f"polya_ratio({d}): no entries with n > 1")
    return max(w * e["upper"] ** d / (e["n"] - 1) for e in sel)


# The thresholds Lambda^*_d = ceil((tau^*_d)^3 d^(3/2)) of Section 5, as
# tabulated in (table:data), for d = 3, ..., 12.
LAMBDA_STAR = {3: 19, 4: 22, 5: 25, 6: 29, 7: 34,
               8: 39, 9: 44, 10: 49, 11: 54, 12: 60}


def gap_table(eps=Fraction(1, 100), ds=range(3, 13), verbose=True):
    """Run the whole verification of Section 5 for the given dimensions and
    return one dict per dimension, holding the exact Polya ratio together with
    the data of Table (table:data).  The full expanded list is kept under the
    key "expanded" for inspection.  Everything except the "(N)" display columns
    is exact.
    """
    eps = Fraction(eps)
    rows = []
    for d in ds:
        Lam = Fraction(LAMBDA_STAR[d])
        list_p = compute_list_p(d, Lam, eps)
        exp = expand_list_p(d, list_p)
        ratio = polya_ratio(d, exp)
        cnt = neumann_counting_function(list_p, d)
        if verbose:
            print(f"d = {d},  Lambda* = {Lam},  K_d = {len(list_p)},  "
                  f"N^Neu = {cnt},  Polya ratio = {float(ratio):.6f},  "
                  f"< 1: {ratio < 1}", flush=True)
        rows.append({"d": d, "Lambda*": Lam, "K": len(list_p), "NNeu": cnt,
                     "ratio": ratio, "ratio(N)": round(float(ratio), 4),
                     "expanded": exp})
    return rows


def polya_check_passed(rows):
    """The certified conclusion: True exactly when (eq:checkpolya) holds in every
    dimension of the run.  The comparison is between exact rationals.
    """
    return all(r["ratio"] < 1 for r in rows)


def gap_table_form(rows, latex=False):
    """Display in the layout of (table:data); with latex=True, emit the LaTeX
    source of the table body (the analogue of TeXForm in the Wolfram package).
    """
    head = ["d", "Lambda*_d", "K_d", "N^Neu", "approx. LHS"]
    body = [[r["d"], r["Lambda*"], r["K"], r["NNeu"], f"{r['ratio(N)']:.4f}"]
            for r in rows]
    if latex:
        lines = [r"\begin{tabular}{rrrrr}", r"\hline",
                 " & ".join(["$d$", r"$\Lambda^*_d$", "$K_d$",
                             r"$\mathcal{N}^{\mathrm{Neu}}$",
                             r"approx. LHS"]) + r" \\", r"\hline"]
        lines += [" & ".join(str(c) for c in row) + r" \\" for row in body]
        lines += [r"\hline", r"\end{tabular}"]
        return "\n".join(lines)
    widths = [max(len(str(h)), max((len(str(row[i])) for row in body), default=0))
              for i, h in enumerate(head)]
    out = ["  ".join(str(h).rjust(widths[i]) for i, h in enumerate(head))]
    out += ["  ".join(str(c).rjust(widths[i]) for i, c in enumerate(row))
            for row in body]
    return "\n".join(out)


# ============================================================================
# Section 7.  DIAGNOSTICS ONLY -- floating-point cross-checks.
# Nothing in this section is used by, or feeds into, any certificate above.
# ============================================================================

def p_prime_float(d, m, k):
    """Float value of p'_{d,m,k}, NON-RIGOROUS, for cross-checking only.
    Requires mpmath.

    For m = 0 the identity p'_{d,0,k} = j_{d/2,k-1} of (lem:pandj) is used.  For
    m >= 1 we bisect  g(x) = m J_nu(x) - x J_{nu+1}(x)  -- the bracketed factor
    of P'_{d,m} in (eq:uprime) -- on the interlacing bracket
    (j_{nu,k-1}, j_{nu,k}) of (eq:pvsj).
    """
    import mpmath as mp
    if m == 0:
        return mp.mpf(0) if k == 1 else mp.besseljzero(mp.mpf(d) / 2, k - 1)
    nu = mp.mpf(m) + mp.mpf(d) / 2 - 1
    g = lambda x: m * mp.besselj(nu, x) - x * mp.besselj(nu + 1, x)
    lo = mp.mpf('1e-6') if k == 1 else mp.besseljzero(nu, k - 1) + mp.mpf('1e-20')
    hi = mp.besseljzero(nu, k) - mp.mpf('1e-20')
    return mp.findroot(g, (lo, hi), solver='bisect', tol=mp.mpf('1e-30'))


def diagnostic_compare(d, Lam, eps, verbose=True):
    """Run the certified algorithm, then check (with floats!) that each returned
    enclosure straddles the float zero.  A smoke test for the IMPLEMENTATION,
    not part of any proof.  Returns (ok, list_p).
    """
    import mpmath as mp
    mp.mp.dps = 30
    list_p = compute_list_p(d, Lam, eps)
    ok = True
    for e in list_p:
        z = p_prime_float(d, e["m"], e["k"])
        lo = mp.mpf(e["lower"].numerator) / e["lower"].denominator
        hi = mp.mpf(e["upper"].numerator) / e["upper"].denominator
        if not (lo <= z <= hi):
            ok = False
            print(f"MISMATCH at (m,k) = ({e['m']},{e['k']}): "
                  f"float {z} not in [{float(lo)}, {float(hi)}]")
    if verbose:
        print(f"{len(list_p)} zeros certified <= {float(Lam)};  "
              f"N^Neu = {neumann_counting_function(list_p, d)};  "
              f"enclosures consistent with floats: {ok}")
    return ok, list_p


# ============================================================================
# Script entry point: reproduce Table (table:data).
# ============================================================================

if __name__ == "__main__":
    import sys
    ds = [int(a) for a in sys.argv[1:]] or list(range(3, 13))
    rows = gap_table(ds=ds)
    print()
    print(gap_table_form(rows))
    print()
    print("Polya check passed in all dimensions of this run:",
          polya_check_passed(rows))
