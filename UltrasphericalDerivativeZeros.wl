(* ::Package:: *)

(* ============================================================================
   UltrasphericalDerivativeZeros.wl

   Certified rational enclosures for the zeros  p'_{d,m,k}  of the derivative
   of the d-dimensional ultraspherical Bessel function of order m,

        P_{d,m}(x) := x^(-(d/2-1)) BesselJ[m + d/2 - 1, x],      d >= 3, m >= 0,

   with the exceptional convention  p'_{d,0,1} := 0.

   This file implements the algorithm of

     "Polya's conjecture for higher-dimensional Neumann balls",
     N. Filonov, M. Levitin, I. Polterovich, D. A. Sher,

   specifically the procedures of the Appendix "Rational enclosure for zeros
   of derivatives of ultraspherical Bessel functions", used to fill the finite
   gap in Section 5 ("Filling the gap").  All numbering below refers to that
   paper: Definition (defn:ratn), Lemmas (lem:enclosures), (lem:trans),
   (lem:pandj), the Lorch-Szego lower bound (eq:LS), and the sign conditions
   (eq:signconditions).

   NOTATION (as in the paper).
        nu   = m + d/2 - 1
        F_nu(t)     = Gamma(nu+1) (x/2)^(-nu) J_nu(x) |_{t = x^2/4}
                    = Sum_j (-t)^j / (j! Pi_j(nu))                (eq:deff)
        H_{d,m}(t)  = m F_{m+d/2-1}(t) - (2t/(m+d/2)) F_{m+d/2}(t) (eq:defh)
        P'_{d,m}(x) = x^(-d/2) (x/2)^(m+d/2-1) / Gamma(m+d/2)
                        * H_{d,m}(x^2/4)                          (eq:uprime)
   Both F_nu and H_{d,m} are entire with RATIONAL Taylor coefficients (nu is a
   half-integer), and H_{d,m}(0) = m.  In the notation of (defn:ratn),
   f_{nu,k} = j_{nu,k}^2/4  are the positive zeros of F_nu, and
   h_{d,m,k} = (p'_{d,m,k})^2/4 those of H_{d,m}.

   MAIN TASK (Section 5).
     ComputeListP[d, Lam, eps]  implements the procedure Main(d, Lambda, eps).
        d    -- integer >= 3;
        Lam  -- rational (or integer) Lambda > 0;
        eps  -- rational (or integer) tolerance epsilon > 0;
     Output: the complete list ListP_{d,Lam} of all pairs (m,k) with
     p'_{d,m,k} <= Lam, returned as associations
        <| "m" -> m, "k" -> k, "lower" -> ., "upper" -> . |>
     where "upper" is the certified rational  \overline{p'_{d,m,k}}  with
        p'_{d,m,k} < \overline{p'_{d,m,k}} <= Min[p'_{d,m,k} + eps, Lam],
     and "lower" is the companion  \underline{p'_{d,m,k}} < p'_{d,m,k}, so that
     the two-sided enclosure of width at most eps mentioned in Section 5 is
     available.  For the exceptional zero the entry (0,1,0) is returned.

     NeumannCountingFunction[listP, d]  returns the exactly computed integer
     N^Neu_{B^d}(Lam) = Sum of the multiplicities kappa_{d,m} over ListP,
     the fourth column of Table (table:data).

   VERIFICATION OF SECTION 5.  Section 6 below assembles the re-indexed list
   of Section 5 (the indices n_i and multiplicities kappa_{d,m}) and evaluates
   the left-hand side of (eq:checkpolya) as an exact rational:
        GapTable[]                 -- runs d = 3, ..., 12 at the tabulated
                                      thresholds Lambda^*_d, returning the
                                      data of Table (table:data);
        PolyaCheckPassed[rows]     -- True iff (eq:checkpolya) holds for all
                                      dimensions of the run;
        GapTableForm[rows]         -- the table in display layout (wrap in
                                      TeXForm for LaTeX).

   DESIGN RULE.  Every certificate rests exclusively on EXACT Integer/Rational
   arithmetic, as required in Section 5: no N[...], no machine reals, and no
   floating-point comparison occurs anywhere in Sections 1-5 below.  Machine
   precision appears ONLY in the clearly marked diagnostic Section 6, which
   feeds into no certificate and exists purely as a cross-check.

   VALIDATION.  This file could not be executed in the environment where it
   was written (no Wolfram kernel).  Its logic was validated by running a
   line-by-line mirror in exact rational arithmetic (Python/Fraction) against
   independent high-precision evaluations of the zeros for
   (d, Lam) = (3,8), (4,10), (5,12), (6,15/2): in each case the certified
   (m,k)-sets and the enclosures agreed exactly.  Run DiagnosticCompare
   (Section 6) as a first smoke test in an actual Wolfram session.

   ============================================================================ *)


(* ============================================================================
   Section 0.  Elementary exact helpers
   ============================================================================ *)

(* nu = m + d/2 - 1, and the quantity m(m+d-2) appearing in (eq:LS).  With d
   an Integer, d/2 is exact (Integer or Rational), so all of these are exact. *)
NuOf[d_Integer, m_Integer]  := m + d/2 - 1;
EtaOf[d_Integer]            := d/2 - 1;
KapOf[d_Integer, m_Integer] := m (m + d - 2);

(* Multiplicity kappa_{d,m} of the spherical harmonics H_{d,m} on S^(d-1);
   the single binomial formula is valid for all m >= 0 (Binomial[a,b] = 0 for
   integers 0 <= a < b), e.g. d = 3 gives 2m + 1.  Used to assemble the
   indices n_i and the counting function of Section 5. *)
KappaMult[d_Integer, m_Integer] :=
  Binomial[m + d - 1, d - 1] - Binomial[m + d - 3, d - 1];

(* A certified RATIONAL lower bound for Sqrt[q], q rational >= 0, with dyadic
   denominator 2^prec.  Floor of an exact algebraic number is exact in
   Mathematica, so the result x0 satisfies x0 <= Sqrt[q] exactly.  Used only
   to enter the zero-free initial interval in BesselZeroBrackets below. *)
RatSqrtLower[q_ /; q >= 0, prec_Integer: 16] :=
  Floor[2^prec Sqrt[q]]/2^prec;

(* Grid step of the march in BesselZeroBrackets.  The paper fixes the value 3,
   admissible because consecutive zeros satisfy j_{nu,k} - j_{nu,k-1} >= Pi > 3
   for nu >= 1/2 (Lorch-Szego, as quoted in the Certification clause of that
   procedure). *)
$GridStep = 3;

(* Safety cap on the loops.  The paper PROVES termination of every loop
   (Certification/Termination clauses of the four procedures); the cap exists
   only to turn a hypothetical implementation bug into a clean error message
   rather than a hang. *)
$MaxIter = 100000;

BugAbort[tag_String] :=
  (Message[ComputeListP::bug, tag]; Abort[]);
ComputeListP::bug =
  "Internal safety cap reached in `1`; this contradicts the proved \
termination and indicates an implementation bug.";

(* Load-time collision guard.  A definition whose name coincides with a
   built-in (Protected) System symbol is silently rejected, after which every
   call dispatches to the built-in -- a failure mode that is hard to diagnose
   from the resulting messages.  (This is exactly what happened when the
   procedure below was named Refine, which is System`Refine.)  The test uses
   Names on strings, so it does not itself create any symbols. *)
With[{clash = Select[
    {"NuOf", "EtaOf", "KapOf", "KappaMult", "RatSqrtLower", "BugAbort",
     "FEnclosure", "HEnclosure", "NStart", "SignF", "SignH",
     "BesselZeroBrackets", "RefineBracket", "ClassifyBracket",
     "DerivativeWindows", "MCutoff", "ComputeListP",
     "NeumannCountingFunction", "WeylConstantBound", "ExpandListP",
     "PolyaRatio", "GapTable", "PolyaCheckPassed", "GapTableForm",
     "PPrimeFloat", "DiagnosticCompare"},
    Names["System`" <> #] =!= {} &]},
  If[clash =!= {},
     Print["FATAL: these names collide with built-in System symbols and \
must be renamed before use: ", clash];
     Abort[]]];


(* ============================================================================
   Section 1.  Certified enclosures of F_nu(t) and H_{d,m}(t)
                (Lemma lem:enclosures)
   ============================================================================ *)

(* FEnclosure[nu, t, n] returns a rational interval {lo, hi} with
       lo <= F_nu(t) <= hi ,
   or $Failed if the hypothesis r_{nu,N}(t) < 1 of (lem:enclosures) is not yet
   met (in which case the caller simply increases n).

   The partial sum F_{nu,N}(t) is accumulated incrementally via the exact term
   recursion  a_0 = 1,  a_{j+1} = a_j (-t)/((j+1)(nu+j+1)),  after which
       tau = |a_{N+1}| = t^(N+1) / ((N+1)! Pi_{N+1}(nu)),
       rho = r_{nu,N}(t) = t / ((N+2)(nu+N+2)),
   and the radius is  delta_{F_{nu,N}}(t) = tau/(1 - rho),  exactly as in
   (lem:enclosures).  Every quantity is Rational. *)
FEnclosure[nu_, t_, n_Integer] := Module[{term = 1, s = 1, j, tau, rho, r},
   Do[
     term = term * (-t)/((j + 1) (nu + j + 1));
     s = s + term,
     {j, 0, n - 1}];                      (* now  s = F_{nu,n},  term = a_n  *)
   tau = Abs[term] * t/((n + 1) (nu + n + 1));          (* = |a_{n+1}|       *)
   rho = t/((n + 2) (nu + n + 2));                      (* = r_{nu,n}(t)     *)
   If[rho >= 1, Return[$Failed]];
   r = tau/(1 - rho);                                   (* = delta_{F_{nu,n}}*)
   {s - r, s + r}];

(* HEnclosure[d, m, t, n]: rational interval containing H_{d,m}(t), obtained
   from the two F-enclosures by exact interval arithmetic.  Since t >= 0 and
   the coefficients m and 2t/(nu+1) = 2t/(m+d/2) are >= 0, midpoints and radii
   combine linearly, giving exactly the H_{d,m,N} and delta_{H_{d,m,N}} of
   (lem:enclosures). *)
HEnclosure[d_Integer, m_Integer, t_, n_Integer] :=
  Module[{nu = NuOf[d, m], e1, e2, c, mid, rad},
   e1 = FEnclosure[nu, t, n];
   e2 = FEnclosure[nu + 1, t, n];
   If[e1 === $Failed || e2 === $Failed, Return[$Failed]];
   c = 2 t/(nu + 1);                                    (* = 2t/(m + d/2)    *)
   mid = m (e1[[1]] + e1[[2]])/2 - c (e2[[1]] + e2[[2]])/2;
   rad = m (e1[[2]] - e1[[1]])/2 + c (e2[[2]] - e2[[1]])/2;
   {mid - rad, mid + rad}];


(* ============================================================================
   Section 2.  Procedure Sign(X, x)
   ============================================================================ *)

(* Starting truncation order.  The paper's workflow sets N = 8; any starting
   value is admissible, and n ~ 2 Sqrt[t] merely starts past the hump of the
   series so that the radius already decays fast.  Ceiling of an exact Sqrt is
   exact, so this stays within rational arithmetic. *)
NStart[t_] := Max[8, Ceiling[2 Sqrt[t]]];

(* SignF[nu, x]    = certified Sign of F_nu(x^2/4)     in {+1, -1};
   SignH[d, m, x]  = certified Sign of H_{d,m}(x^2/4)  in {+1, -1}.

   These are the two instantiations X in {F_nu, H_{d,m}} of the procedure
   Sign(X, x).  Both take the point in the variable x (rational, >= 0) and
   query the series at t = x^2/4, again rational.  At x = 0 the exact values
   F_nu(0) = 1 and H_{d,m}(0) = m are used instead of an enclosure; the latter
   is the exception noted in the Input clause of the procedure Refine.

   Doubling loop as in the workflow: compute the enclosure; if it excludes 0,
   return the common sign of its endpoints; otherwise double N.
   Certification: by (lem:enclosures).
   Termination: by (lem:trans), all zeros of F_nu and H_{d,m} are irrational,
   so the true value X(x) is nonzero at rational x and the loop must exit. *)
SignF[nu_, x_] := Module[{t, n, enc, res},
   If[x == 0, Return[1]];                              (* F_nu(0) = 1       *)
   t = x^2/4; n = NStart[t];
   res = Catch[
     Do[
       enc = FEnclosure[nu, t, n];
       If[enc =!= $Failed && (enc[[1]] > 0 || enc[[2]] < 0),
          Throw[Sign[enc[[1]]]]];
       n = 2 n,
       {$MaxIter}];
     $Failed];
   If[res === $Failed, BugAbort["SignF"]];
   res];

SignH[d_Integer, m_Integer, x_] := Module[{t, n, enc, res},
   If[x == 0, Return[Sign[m]]];                        (* H_{d,m}(0) = m    *)
   t = x^2/4; n = NStart[t];
   res = Catch[
     Do[
       enc = HEnclosure[d, m, t, n];
       If[enc =!= $Failed && (enc[[1]] > 0 || enc[[2]] < 0),
          Throw[Sign[enc[[1]]]]];
       n = 2 n,
       {$MaxIter}];
     $Failed];
   If[res === $Failed, BugAbort["SignH"]];
   res];


(* ============================================================================
   Section 3.  Procedure BesselZeroBrackets(nu, Lambda)
   ============================================================================ *)

(* Returns the list
       { {a_1, b_1, sa_1, sb_1}, ..., {a_K, b_K, sa_K, sb_K} }
   of rational brackets  j_{nu,k} in (a_k, b_k),  b_k - a_k = 3,  a_K >= Lam,
   together with the certified signs sa_k, sb_k of F_nu at the two endpoints
   (cached for the later bisections).

   Certification (as in the paper): consecutive zeros satisfy
   j_{nu,k} - j_{nu,k-1} >= Pi > 3 for nu >= 1/2 (Lorch-Szego), so each
   interval of length 3 contains at most one zero, which is simple; the march
   starts inside the zero-free initial interval, since j_{nu,1} > nu.  Rational
   grid points are never zeros by (lem:trans), so all signs are genuine.
   Hence the recorded brackets enumerate, in increasing order and without
   omission, the zeros j_{nu,1}, j_{nu,2}, ... .
   Termination: there are finitely many zeros in (0, Lam].

   NOTE ON THE ENTRY POINT.  The paper's workflow starts at x_0 = nu.  The
   code starts instead at a rational lower bound for Sqrt[nu^2 - 1/4] <= nu,
   which lies in the same zero-free interval and is therefore equally valid,
   merely taking a few more grid steps.  To match the paper's workflow exactly,
   replace the line defining x0 by   x0 = nu;   (nu is a half-integer, hence
   exactly rational, so this stays within exact arithmetic). *)
BesselZeroBrackets[nu_, Lam_] := Module[
   {x0, xi, si, xn, sn, brackets = {}, res},
   x0 = If[nu == 1/2, 0, RatSqrtLower[nu^2 - 1/4]];
   xi = x0; si = SignF[nu, xi];           (* = +1, as j_{nu,1} > nu >= x0   *)
   res = Catch[
     Do[
       xn = xi + $GridStep;
       sn = SignF[nu, xn];
       If[si sn < 0,                      (* certified sign change:         *)
          AppendTo[brackets, {xi, xn, si, sn}];  (* exactly one zero here   *)
          If[xi >= Lam, Throw[brackets]]];       (* stop: a_K >= Lambda     *)
       xi = xn; si = sn,
       {$MaxIter}];
     $Failed];
   If[res === $Failed, BugAbort["BesselZeroBrackets"]];
   res];


(* ============================================================================
   Section 4.  Procedure Refine(X, (a,b))  [code name: RefineBracket]
   ============================================================================ *)

(* One bisection step on a bracket {a, b, sa, sb} carrying a certified sign
   change of the function whose sign oracle is signFun (a pure function
   x |-> +-1); at a zero left endpoint the exact value H_{d,m}(0) = m is used
   in place of Sign(X, 0), as stated in the Input clause of the procedure.
   The rational midpoint is never a zero by (lem:trans), so the returned
   half-width bracket again carries a certified sign change and contains the
   same unique zero.

   NAME.  The paper calls this procedure Refine.  That name CANNOT be used
   here: Refine is a built-in Wolfram symbol (Refine[expr, assumptions]) and
   is Protected, so a definition of it is silently rejected and every call
   dispatches to the built-in.  We therefore use RefineBracket. *)
RefineBracket[signFun_, {l_, r_, sl_, sr_}] := Module[{mid, sm},
   mid = (l + r)/2;                (* dyadic-friendly exact midpoint         *)
   sm = signFun[mid];
   If[sm == sl, {mid, r, sm, sr}, {l, mid, sl, sm}]];


(* ============================================================================
   Section 5.  Procedure Main(d, Lambda, eps)
   ============================================================================ *)

(* ClassifyBracket implements the two exit conditions (i) / (ii) shared by
   step 1 and step 2.3 of Main.  Given a bracket certified to contain a unique
   zero z of the function with sign oracle signFun, it bisects until either
     (i)  b <= Lam and b - a <= eps: return {"include", a, b}, the final
          two-sided enclosure a < z < b <= Min[Lam, z + eps]; or
     (ii) a >= Lam: return {"exclude"}.
   Exactly one exit is reached after finitely many steps: by (lem:trans),
   z != Lam, so z < Lam forces (i) and z > Lam forces (ii). *)
ClassifyBracket[signFun_, br0_List, Lam_, eps_] := Module[{br = br0, res},
   res = Catch[
     Do[
       Which[
         br[[2]] <= Lam && br[[2]] - br[[1]] <= eps,
           Throw[{"include", br[[1]], br[[2]]}],
         br[[1]] >= Lam,
           Throw[{"exclude"}],
         True,
           br = RefineBracket[signFun, br]],
       {$MaxIter}];
     $Failed];
   If[res === $Failed, BugAbort["ClassifyBracket"]];
   res];

(* DerivativeWindows implements steps 2.1 and 2.2 of Main for m >= 1.

   Step 2.1: refine each bracket (a_k, b_k) of BesselZeroBrackets, bisecting
   on F_nu, until the ultraspherical derivative sign conditions
        Sign(H_{d,m}, a_k) = Sign(H_{d,m}, b_k) = (-1)^k          (eq:signconditions)
   hold.  Step termination: as a_k increases to j_{nu,k} and b_k decreases to
   j_{nu,k}, both signs eventually equal (-1)^k by (lem:pandj), which gives
   sign P'_{d,m}(j_{nu,k}) = (-1)^k; each RefineBracket halves the width.  Refining
   preserves a_K >= Lam, since a_K can only increase.

   Step 2.2: return the ultraspherical derivative windows
        W_0 = (0, a_1),      W_k = (b_k, a_{k+1}),   k = 1, ..., K-1,
   each as a bracket-with-signs for the H_{d,m} oracle; at the left endpoint of
   W_0 the exact value H_{d,m}(0) = m > 0 is used.  By (lem:pandj) and
   (eq:signconditions), W_k contains exactly one zero of H_{d,m}, namely
   p'_{d,m,k+1}; no zero of H_{d,m} in (0, j_{nu,K}) lies outside the closure
   of the union of the W_k, while p'_{d,m,K+1} > j_{nu,K} > a_K >= Lam.

   Subtle case handled automatically: consecutive zeros j_{nu,k} may fall in
   ADJACENT grid cells (their gap can be as small as Pi > 3), so that initially
   b_k = a_{k+1}.  The sign conditions at that shared point would be
   contradictory ((-1)^k versus (-1)^(k+1)), so the conditioning loop
   necessarily keeps refining until b_k and a_{k+1} separate; the certified
   signs then force b_k < p'_{d,m,k+1} < a_{k+1}, a nonempty window. *)
DerivativeWindows[d_Integer, m_Integer, dirBrs_List] := Module[
   {nu = NuOf[d, m], K = Length[dirBrs], brs = dirBrs,
    target, k, windows, done},
   (* -- Step 2.1: enforce the sign conditions (eq:signconditions) -------- *)
   Do[
     target = (-1)^k;
     done = Catch[
       Do[
         If[SignH[d, m, brs[[k, 1]]] == target &&
            SignH[d, m, brs[[k, 2]]] == target,
            Throw[True]];
         brs[[k]] = RefineBracket[SignF[nu, #] &, brs[[k]]],
         {$MaxIter}];
       False];
     If[done === False, BugAbort["DerivativeWindows"]],
     {k, K}];
   (* -- Step 2.2: assemble the windows W_0, ..., W_{K-1} ----------------- *)
   windows = Table[
     If[k == 0,
        {0, brs[[1, 1]], 1, -1},
        {brs[[k, 2]], brs[[k + 1, 1]], (-1)^k, (-1)^(k + 1)}],
     {k, 0, K - 1}];
   windows];

(* The cut-off M_{d,Lambda} = Max{ m >= 0 : m(m + d - 2) < Lambda^2 } quoted in
   Main, justified by the Lorch-Szego lower bound (eq:LS),
   p'_{d,m,1} > Sqrt(m(m+d-2)) for m >= 1.  Exact integer march. *)
MCutoff[d_Integer, Lam_] := Module[{M = 0},
   While[(M + 1) (M + d - 1) < Lam^2, M++]; M];

(* ---------------------------------------------------------------------- *)
(* ComputeListP[d, Lam, eps] -- the procedure Main; see the header for the *)
(* output contract (the list ListP of Section 5).                          *)
(* ---------------------------------------------------------------------- *)
ComputeListP[d_Integer /; d >= 3,
             Lam_ /; Element[Lam, Rationals] && Lam > 0,
             eps_ /; Element[eps, Rationals] && eps > 0] :=
 Module[{L = {}, nu, brs, wins, res, M, m, k},

  (* ---- Step 0: initialise with the exceptional zero p'_{d,0,1} = 0 ---- *)
  AppendTo[L, <|"m" -> 0, "k" -> 1, "lower" -> 0, "upper" -> 0|>];

  (* ---- Step 1: the case m = 0 ----------------------------------------
     By (lem:pandj), p'_{d,0,k} = j_{d/2,k-1}, so the zeros are classified
     directly on the brackets for nu = d/2 using the F-oracle.  They increase
     in k, so we may stop at the first exclusion; by construction a_K >= Lam,
     so exit (ii) occurs at the latest at k = K, and all zeros with k > K
     satisfy j_{d/2,k} > j_{d/2,K} > a_K >= Lam. ---------------------- *)
  nu = d/2;
  brs = BesselZeroBrackets[nu, Lam];
  Do[
    res = ClassifyBracket[SignF[nu, #] &, brs[[k]], Lam, eps];
    If[res[[1]] === "include",
       AppendTo[L, <|"m" -> 0, "k" -> k + 1,
                     "lower" -> res[[2]], "upper" -> res[[3]]|>],
       Break[]],
    {k, Length[brs]}];

  (* ---- Step 2: the cases 1 <= m <= M_{d,Lambda} ----------------------- *)
  M = MCutoff[d, Lam];
  Do[
    nu = NuOf[d, m];
    brs = BesselZeroBrackets[nu, Lam];      (* exhaustive j_{nu,k}, k <= K  *)
    wins = DerivativeWindows[d, m, brs];    (* W_0, ..., W_{K-1}           *)
    Do[
      (* window W_{k-1} contains exactly one zero of H_{d,m}: p'_{d,m,k}   *)
      res = ClassifyBracket[SignH[d, m, #] &, wins[[k]], Lam, eps];
      If[res[[1]] === "include",
         AppendTo[L, <|"m" -> m, "k" -> k,
                       "lower" -> res[[2]], "upper" -> res[[3]]|>],
         Break[]],                          (* zeros increase in k          *)
      {k, Length[wins]}],
    {m, 1, M}];

  (* ---- Step 3: return ListP ------------------------------------------ *)
  SortBy[L, {#["m"] &, #["k"] &}]];

(* Exact value of the Neumann counting function of the unit ball at Lambda,
   the fourth column of Table (table:data):
       N^Neu_{B^d}(Lam) = Sum over (m,k) in ListP of kappa_{d,m}.          *)
NeumannCountingFunction[result_List, d_Integer] :=
  Total[KappaMult[d, #["m"]] & /@ result];


(* ============================================================================
   Section 6.  Assembling the table of Section 5 and checking Polya's
               conjecture, i.e. the inequality (eq:checkpolya)
   ============================================================================ *)

(* A rational UPPER bound for the Weyl constant

        w_d = 2^(-d) / Gamma(1 + d/2)^2

   (the constant of Weyl's law for the unit ball, since |B^d| = omega_d and
   w_d = omega_d^2/(2 Pi)^d).  For even d this is already rational.  For odd d
   a factor 1/Pi remains, and, exactly as stated in Section 5, we replace 1/Pi
   by 1/3; since Pi > 3 this OVERSTATES w_d, so the resulting ratio overstates
   the left-hand side of (eq:checkpolya) and the check stays rigorous.

   Implementation note: multiplying by Pi clears the factor for odd d, and the
   result is then asserted to be rational before it is returned -- so a silent
   failure to reduce cannot slip through into a "certified" number. *)
WeylConstantBound[d_Integer] := Module[{w, r},
   w = 2^-d/Gamma[1 + d/2]^2;
   r = If[OddQ[d], (w Pi)/3, w];          (* w Pi is rational for odd d      *)
   If[! (Head[r] === Integer || Head[r] === Rational),
      Print["FATAL: WeylConstantBound[", d, "] did not reduce to a rational: ",
            r]; Abort[]];
   r];

(* ExpandListP re-arranges and re-indexes ListP exactly as in Section 5:
   the entries are sorted by their certified upper bounds and given the
   cumulative indices
        n_1 = 1,     n_{i+1} = n_i + kappa_{d,m_i},
   so that the Neumann eigenvalues of B^d satisfy
        mu_{n_i} = ... = mu_{n_i + kappa_{d,m_i} - 1} < (upper_i)^2.
   The exceptional entry (0,1,0) has upper bound 0 and therefore sorts first,
   giving (n_1, m_1, k_1) = (1, 0, 1) as required.

   Why sorting by the UPPER bounds is legitimate even though the enclosures
   may in principle overlap: every entry placed before position i has upper
   bound <= upper_i, hence its true zero is < upper_i.  So the n_i - 1
   eigenvalues accumulated before position i are all < (upper_i)^2 whatever
   the true order of two nearly equal zeros happens to be, and the displayed
   inequality for mu_{n_i} holds regardless.  Ties are broken by (m,k) purely
   for reproducibility.

   The exact rationals are retained in "lower"/"upper"; "lower(N)"/"upper(N)"
   are decimal approximations for display only and enter no certificate. *)
ExpandListP[d_Integer, listP_List, digits_Integer : 6] :=
 Module[{s, out, n = 1, kap, e, j},
   s = SortBy[listP, {#["upper"] &, #["m"] &, #["k"] &}];
   out = ConstantArray[Null, Length[s]];
   Do[
     e = s[[j]];
     kap = KappaMult[d, e["m"]];
     out[[j]] = <|"n" -> n, "m" -> e["m"], "k" -> e["k"],
                  "multiplicity" -> kap,
                  "lower" -> e["lower"], "upper" -> e["upper"],
                  "lower(N)" -> N[e["lower"], digits],
                  "upper(N)" -> N[e["upper"], digits]|>;
     n = n + kap,
     {j, Length[s]}];
   out];

(* PolyaRatio returns the left-hand side of (eq:checkpolya),

        max_{i >= 2}  w_d (upper_i)^d / (n_i - 1) ,

   as an EXACT rational number (all inputs are rational, w_d being the bound
   above).  The first entry, with n_1 = 1, is excluded: it would divide by
   n_1 - 1 = 0.  Polya's conjecture holds on the range covered by ListP
   precisely when this value is < 1. *)
PolyaRatio[d_Integer, expanded_List] :=
 Module[{w = WeylConstantBound[d], sel},
   sel = Select[expanded, #["n"] > 1 &];
   If[sel === {},
      Print["FATAL: PolyaRatio[", d, "]: no entries with n > 1."]; Abort[]];
   Max[w #["upper"]^d/(#["n"] - 1) & /@ sel]];

(* The thresholds Lambda^*_d = Ceiling[(tau^*_d)^3 d^(3/2)] of Section 5,
   as tabulated in (table:data), for d = 3, ..., 12. *)
$LambdaStar = <|3 -> 19, 4 -> 22, 5 -> 25, 6 -> 29, 7 -> 34,
                8 -> 39, 9 -> 44, 10 -> 49, 11 -> 54, 12 -> 60|>;

(* GapTable runs the whole verification of Section 5 for the given dimensions
   and returns one association per dimension, holding the exact Polya ratio
   together with the data of (table:data).  The full expanded list is kept
   under the key "expanded" for inspection.
   Everything except the "..." (N) display columns is exact. *)
GapTable[eps_ : 1/100, ds_ : Range[3, 12], verbose_ : True] :=
 Module[{rows = {}, listP, exp, ratio, Lam, d, cnt},
   Do[
     Lam = $LambdaStar[d];
     listP = ComputeListP[d, Lam, eps];
     exp = ExpandListP[d, listP];
     ratio = PolyaRatio[d, exp];
     cnt = NeumannCountingFunction[listP, d];
     If[verbose,
        Print["d = ", d, ",  Lambda* = ", Lam, ",  K_d = ", Length[listP],
              ",  N^Neu = ", cnt, ",  Polya ratio = ", N[ratio, 6],
              ",  < 1: ", ratio < 1]];
     AppendTo[rows,
       <|"d" -> d, "Lambda*" -> Lam, "K" -> Length[listP], "NNeu" -> cnt,
         "ratio" -> ratio, "ratio(N)" -> N[ratio, 4], "expanded" -> exp|>],
     {d, ds}];
   rows];

(* The certified conclusion: True exactly when (eq:checkpolya) holds in every
   dimension of the run.  The comparison is between exact rationals. *)
PolyaCheckPassed[rows_List] := AllTrue[rows, #["ratio"] < 1 &];

(* Display in the layout of (table:data); wrap in TeXForm for the LaTeX
   source of the table (note: TeXForm, not TexForm). *)
GapTableForm[rows_List] :=
  TableForm[
    {#["d"], #["Lambda*"], #["K"], #["NNeu"], #["ratio(N)"]} & /@ rows,
    TableHeadings -> {None,
      {"d", "\[CapitalLambda]*_d", "K_d", "N^Neu", "approx. LHS"}}];


(* ============================================================================
   Section 7.  DIAGNOSTICS ONLY -- machine-precision cross-checks.
   Nothing in this section is used by, or feeds into, any certificate above.
   ============================================================================ *)

(* Float value of p'_{d,m,k}, NON-RIGOROUS, for cross-checking only.

   For m = 0 the identity p'_{d,0,k} = j_{d/2,k-1} of (lem:pandj) is used.  For
   m >= 1 we bisect  g(x) = m J_nu(x) - x J_{nu+1}(x)  -- the bracketed factor
   of P'_{d,m} in the paper's displayed formula for the derivative -- on the
   interlacing bracket (j_{nu,k-1}, j_{nu,k}) of (eq:pvsj), whose endpoint
   signs are (-1)^(k-1) and (-1)^k by (lem:pandj).  We START from the known
   left-end sign and never evaluate g at the endpoints themselves (near x = 0
   the value m J_nu(10^-6) underflows at machine precision).  Evaluation is at
   30 significant digits; 60 bisection steps on an interval of length <~ 2 Pi
   give ~15+ digits, ample for a plausibility check. *)
PPrimeFloat[d_Integer, m_Integer, k_Integer] := Module[
   {nu, g, lo, hi, signLo, mid, wp = 30},
   If[m == 0,
      If[k == 1, Return[0],
         Return[N[BesselJZero[d/2, k - 1], wp]]]];
   nu = NuOf[d, m];                                    (* exact half-integer *)
   g[x_] := m BesselJ[nu, x] - x BesselJ[nu + 1, x];
   lo = If[k == 1, N[10^-6, wp],
           N[BesselJZero[nu, k - 1], wp] + 10^-20];
   hi = N[BesselJZero[nu, k], wp] - 10^-20;
   signLo = (-1)^(k - 1);          (* sign of g just right of j_{nu,k-1}    *)
   Do[
     mid = (lo + hi)/2;
     If[Sign[g[mid]] == signLo, lo = mid, hi = mid],
     {60}];
   (lo + hi)/2];

(* DiagnosticCompare[d, Lam, eps]: runs the certified algorithm, then checks
   (with floats!) that each returned enclosure indeed straddles the float zero.
   A non-numeric float value is itself reported as a failure of the DIAGNOSTIC
   (never of the certificate).  Smoke test for the IMPLEMENTATION, not part of
   any proof. *)
DiagnosticCompare[d_Integer, Lam_, eps_] := Module[
   {L, ok = True, z},
   L = ComputeListP[d, Lam, eps];
   Scan[
     (z = PPrimeFloat[d, #["m"], #["k"]];
      Which[
        !NumericQ[z],
          ok = False;
          Print["DIAGNOSTIC FAILURE at (m,k) = ", {#["m"], #["k"]},
                ": float solver returned ", z],
        !(N[#["lower"], 30] <= z <= N[#["upper"], 30]),
          ok = False;
          Print["MISMATCH at (m,k) = ", {#["m"], #["k"]},
                ": float ", N[z], " not in ",
                N[{#["lower"], #["upper"]}]]]) &, L];
   Print[Length[L], " zeros certified <= ", N[Lam],
         ";  N^Neu = ", NeumannCountingFunction[L, d],
         ";  enclosures consistent with floats: ", ok];
   {ok, L}];

(* Example usage (uncomment to run):

   (* -- a single dimension, reproducing one row of Table (table:data): *)

     listP = ComputeListP[3, 19, 1/100];        (* d = 3, Lambda^*_3 = 19  *)
     Length[listP]                              (* K_3 = 50                *)
     NeumannCountingFunction[listP, 3]          (* N^Neu = 570             *)
     Dataset[ExpandListP[3, listP]]             (* the expanded list       *)
     PolyaRatio[3, ExpandListP[3, listP]]       (* exact; approx 0.9061    *)

  (*  -- the full verification of Section 5, d = 3, ..., 12: *)

     rows = GapTable[];                         (* prints progress per d   *)
     PolyaCheckPassed[rows]                     (* must be True            *)
     GapTableForm[rows]
     TeXForm[GapTableForm[rows]]                (* LaTeX source            *)

   (* -- optional float cross-check of the enclosures in one dimension: *)

     DiagnosticCompare[5, 25, 1/100];

*)
