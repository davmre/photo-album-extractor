import numpy as np
import scipy.optimize

from core import geometry
from core.photo_types import BoundingBoxAny, QuadArray, bounding_box_as_array

"""
Tools to compute the largest rectangle inscribable in a (convex) quadrilateral.

While finding the minimum bounding rectagle (the "outer" rectangle) is trivial,
finding the maximum "inner" rectangle is quite nontrivial. The approach
taken in this file is to observe that the maximum inner rectangle always
contacts the quadrilateral at at least three points (it is straightforward to
show that otherwise we always have the freedom to make it bigger, which is a
contradiction). Thus we can break the problem into cases according to
the edges along which those three contact points lie.

The class `InscriptionGeometry` represents the situation in which we have
selected three contact edges, along with the location of a contact
point along the first edge (the 'pivot' point). Given this information it
is possible to solve for the largest inscribable rectangle that contacts
at the pivot point.

For each edge configuration, we therefore need to optimize over all pivot
points (represented by a scalar parameter giving its location along the edge).
This is done first via a coarse grid search to identify the neighborhood
of the global optimum (there may be multiple local optima), and then
minimized in this neighborhood using scipy.optimize.minimize_scalar
(bounded Brent's method).
"""


def cross2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = np.asarray(x), np.asarray(y)
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def merge_adjacent_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged = []
    for a, b in intervals:
        if not merged or merged[-1][1] < a:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    return merged


def minimize_scalar_multimodal(fn, bounds, n_eval_points):
    xs = np.linspace(bounds[0], bounds[1], n_eval_points)
    fs = [fn(x) for x in xs]
    best_idx = np.argmin(fs)
    x_lower = xs[best_idx - 1] if best_idx > 0 else bounds[0]
    x_upper = xs[best_idx + 1] if best_idx < n_eval_points - 1 else bounds[1]
    return scipy.optimize.minimize_scalar(
        fn, bounds=[x_lower, x_upper], method="bounded"
    )


class GeometryNotFeasibleError(Exception):
    pass


def get_real_roots_of_quadratic_equation(A, B, C):
    """Computes real values t where `A * t**2 + B * t + C = 0`."""
    if A == 0:  # We actually just have a linear equation Bt + C = 0.
        if B == 0:
            return []
        return [-C / B]
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:  # No real roots.
        return []
    elif discriminant == 0:  # One real root.
        return [-B / (2 * A)]
    else:  # Two real roots
        return [
            (-B + np.sqrt(discriminant)) / (2 * A),
            (-B - np.sqrt(discriminant)) / (2 * A),
        ]


class InscriptionGeometry:
    """
  Assume we are given a quadrilateral with four points in counterclockwise
  order. We are considering rectangles inscribed in this quadrilateral that
  contact it at (at least) three points p1, p2, p3. We begin with a fixed
  'pivot' point `p1 = (1 - t1) * B + t1 * C` (this pivot point, along with the
  set of contact edges, will be optimized in an outer loop). We are interested
  in solving for the largest-area rectangle that contacts the given three
  contact edges.

  ### Setup

  The picture looks something like this:

  D----------A
  |           \
  |            (p2)
  (p3)          \
  |              \
  C-----(p1)------B

  The notation is worked out assuming that point p2 lies on the edge AB
  (`p2 = (1 - t2) * A + t2 * B` for some parameter t2)
  and similarly that p3 lies on the edge CD, as seen above. This is generally
  the right picture to hold in your head, and it turns out that the same math
  works even when we need to consider some weirder cases that
  don't quite match this picture.
  
  For example, in practice we may consider contacts where these edges are in
  fact adjacent (e.g., some vertex effectively serves as both point A and D):

  C--(p3)------D/A
  |             \
  |            (p2)
  |              \
  |               \
   -----(p1)------B

  and we may also have the case where p1 and p2 are in fact on the same edge:

  C--(p3)---------D
  |                \
  |                 \
  |                  \
  |                   \
  B--(p2)-----(p1)----A

  We would refer to this edge as BC if parameterizing p1, and AB if
  parameterizing p2. But since p1 is fixed, it causes no problems to exclusively
  refer to this edge as AB since all our calculations will be in reference to
  p2.

  ### Strategy
  
  Given our pivot point p1, we only need to choose the second point p2
  (parameterized by the scalar t2) in order to completely
  specify an inscribed rectangle. Once we have the segment (p1, p2) as a side
  of the rectangle, we extend a perpendicular segment from p1 until it contacts
  the third edge of the bounding quadrilateral, at some point p3. The segments
  (p1, p2) and (p1, p3) thus define both sides of the inscribed rectangle, and
  we can trivially compute the final corner `p4 = p1 + (p2 - p1) + (p3 - p1)`.

  The straightforward way to choose the optimal `t2` is to compute the area
  of the bounding rectangle
  || p2 - p1 || || p3 - p1 ||
  solve for the critical points where the derivative is zero. The derivative
  turns out to be a quadratic equation with two potential zeros; these are
  the potential optima for t2. See the math worked out by o3 here:
  https://chatgpt.com/share/68523c6c-df9c-8003-93eb-98941380b00d

  For any given choice of contact edges and contact point `t2`, one of a
  couple of issues might come up:

  1. The segment perpendicular to (p2 - p1) may not actually contact the 
     edge we have selected for the third contact point. For example, in this
     setup

     -(p3)--------A
     |            \
     |             \
     |              \
     |             (p2)
     ------(p1)------B

     the perpendicular segment to (p2 - p1) will intersect the top edge of the
     quadrilateral. If we had instead specified the left edge (the vertical
     line) as our third contact edge, we would find an intersection point
     that lies outside the quadrilateral:
     
     (p3)
     |
     -----------A
     |           \
     |            \
     |             \
     |             (p2)
     ------(p1)------B

     This corresponds to a parameter t3 lying outside of the interval [0, 1].

  2. We may have three valid contact points p1, p2, and p3, but then find that
     the implied fourth point `p4` lies outside the quadrilateral:
 
               (p4)
     -(p3)----A
     |         \
     |          \
     |           \
     |           (p2)
     ------(p1)----B
  
    We can see this as the point `p4` violating at least one of the four
    half-plane constraints imposed by the sides of the bounding quadrilateral.

  We need to back up these constraints into a feasible region (s) for
  t2. The optimal t2 will thus lie either at a critical point of the area
  function *or* at a boundary point of the feasible region. Thus to find the
  optimal t2, we check all of these possibilities and simply choose the one with
  the largest area.

  """

    def __init__(
        self,
        quad_counterclockwise,
        contact_edge1_idx,
        contact_edge2_idx,
        contact_edge3_idx,
        t1,
    ):
        # Note that counterclockwise in Cartesian coordinates (used here) is equivalent
        # to clockwise in image coordinates (used elsewhere in this codebase). We
        # require this ordering convention so that the normal vectors to edges
        # always point inwards, but we don't explicitly check the condition here
        # because we assume it is enforced by an outer-loop optimizer.
        quad = np.array(quad_counterclockwise)
        self.quad = quad

        x1, x2, x3, x4 = [np.asarray(p) for p in quad]
        self.edges = [(x1, x2), (x2, x3), (x3, x4), (x4, x1)]

        self.contact_edge1_idx = contact_edge1_idx
        self.contact_edge2_idx = contact_edge2_idx
        self.contact_edge3_idx = contact_edge3_idx

        # Compute the first contact point p1.
        contact_edge1 = self.edges[contact_edge1_idx]
        self.contact_point_1 = (1 - t1) * contact_edge1[0] + t1 * contact_edge1[1]

        # The points A,B and C,D are defined as the endpoints of the two remaining
        # contact edges.
        self.pt_a, self.pt_b = self.edges[contact_edge2_idx]
        self.pt_c, self.pt_d = self.edges[contact_edge3_idx]

        # Precompute some useful quantities.
        self.cp_vec = self.pt_c - self.contact_point_1
        self.ap_vec = self.pt_a - self.contact_point_1
        self.ba_vec = self.pt_b - self.pt_a  # aka edge2_vec
        self.dp_vec = self.pt_d - self.contact_point_1
        self.dc_vec = self.pt_d - self.pt_c  # aka edge3_vec

        self.A = np.dot(self.ap_vec, self.ap_vec)
        self.B = np.dot(self.ap_vec, self.ba_vec)
        self.C = np.dot(self.ba_vec, self.ba_vec)
        self.D = np.dot(self.ap_vec, self.dc_vec)
        self.E = np.dot(self.ba_vec, self.dc_vec)

        self.F = -np.dot(self.cp_vec, self.ap_vec)
        self.G = -np.dot(self.cp_vec, self.ba_vec)
        self.K = cross2d(self.cp_vec, self.dc_vec)

    def get_t2_cutpoints_from_third_edge_bounds(self) -> list[float]:
        cutpoints = []
        if self.contact_edge1_idx != self.contact_edge2_idx:
            if self.G != 0:
                t2_at_t3_is_zero = (
                    -self.F / self.G
                )  # np.dot(g.ap_vec, g.cp_vec) / ( np.dot(g.ba_vec, g.cp_vec))
                cutpoints.append(t2_at_t3_is_zero)

            t2_at_t3_is_one_denominator = np.dot(self.ba_vec, self.dp_vec)
            if t2_at_t3_is_one_denominator != 0:
                t2_at_t3_is_one = (
                    -np.dot(self.ap_vec, self.dp_vec) / t2_at_t3_is_one_denominator
                )
                cutpoints.append(t2_at_t3_is_one)
        return cutpoints

    def get_t2_cutpoints_from_edge_halfplane_constraints(self) -> list[float]:
        cutpoints = []
        for X, Y in self.edges:  # Each edge contributes a halfplane constraint to p4.
            edge = Y - X

            A_i = cross2d(edge, self.pt_a - X)
            B_i = cross2d(edge, self.ba_vec)
            C_i = -np.dot(edge, self.ap_vec)
            D_i = -np.dot(edge, self.ba_vec)

            # The halfplane constraint for each edge $i$ looks like
            # $h(t2) = (A_i + B_i t2)(self.D + self.E t2) - self.K (C_i + D_i t2) <= 0$.
            # We can rearrange this as a quadratic equation in t2, of the form
            # (quad_A) t2**2 + (quad_B) t2 + quad_C with the following coefficients:
            quad_A = B_i * self.E
            quad_B = B_i * self.D + A_i * self.E + self.K * D_i
            quad_C = A_i * self.D + self.K * C_i

            # The sign of this quantity corresponds to which side of edge $i$ our
            # point p4 lies. So we solve for the zeros, as the locations where the
            # sign may change.
            cutpoints += get_real_roots_of_quadratic_equation(quad_A, quad_B, quad_C)

        return cutpoints

    def get_t2_cutpoints(self):
        """Compute points where t2 may switch between feasible and infeasible."""
        cutpoints = []
        cutpoints += self.get_t2_cutpoints_from_third_edge_bounds()
        cutpoints += self.get_t2_cutpoints_from_edge_halfplane_constraints()
        cutpoints = (
            [0.0] + sorted({float(x) for x in cutpoints if x >= 0 and x <= 1}) + [1.0]
        )
        return cutpoints

    def get_feasible_t2s(self, sorted_cutpoints=None, eps=1e-10):
        if sorted_cutpoints is None:
            sorted_cutpoints = self.get_t2_cutpoints()
        valid_intervals = []
        for i in range(len(sorted_cutpoints) - 1):
            # For each pair of adjacent cutpoints, test a candidate t2.
            test_t2 = (sorted_cutpoints[i] + sorted_cutpoints[i + 1]) / 2

            # Check that we hit the third contact edge at t3 in [0, 1].
            # alpha = -g.K / (g.D + g.E * test_t2)
            # print(f"alpha at t2={test_t2}: {alpha}")
            t3_denominator = self.D + self.E * test_t2
            if np.abs(t3_denominator) <= eps:
                continue
            t3 = (self.F + self.G * test_t2) / t3_denominator

            # print(
            #    f"t3 at t2={test_t2}: {t3} from num {(self.F + self.G * test_t2)} denom {t3_denominator}"
            # )
            if t3 < 0 or t3 > 1:
                continue

            # Then test the halfplane constraints on `p4` for all four edges.
            contact_point_2 = (1 - test_t2) * self.pt_a + test_t2 * self.pt_b
            contact_point_3 = (1 - t3) * self.pt_c + t3 * self.pt_d
            point4 = (
                self.contact_point_1
                + (contact_point_2 - self.contact_point_1)
                + (contact_point_3 - self.contact_point_1)
            )

            is_valid = True
            for X, Y in self.edges:
                edge = Y - X
                cond = cross2d(edge, point4 - X)

                # equiv_cond = A_i + B_i * test_t2 - alpha * (C_i + D_i * test_t2)
                # cond_h = (A_i + B_i * test_t2) * (self.D + self.E * test_t2) + self.K * (C_i + D_i * test_t2)
                # DEt = self.D + self.E * test_t2
                # print(f"cond {cond} equiv {equiv_cond} cond_h {cond_h} DEt {DEt} cond_h_equiv {cond_h / DEt}")

                # print(f"edge {X} {Y} tests {edge} x {point4} - X = {cond}")
                if cond < 0:
                    is_valid = False
                    break
            if is_valid:
                valid_intervals.append((sorted_cutpoints[i], sorted_cutpoints[i + 1]))

        # Merge redundant cases like `[(0, 0.4), (0.4, 0.6)]` => `[(0, 0.6)]`.
        valid_intervals = merge_adjacent_intervals(valid_intervals)

        # print("valid intervals", valid_intervals)
        return valid_intervals

    def critical_points_of_area_wrt_t2(self):
        quad_A = self.C * self.E
        quad_B = 2 * self.C * self.D
        quad_C = 2 * self.B * self.D - self.E * self.A
        return get_real_roots_of_quadratic_equation(quad_A, quad_B, quad_C)

    def rect_from_t2(self, t2, eps=1e-8):
        contact_point_2 = (1 - t2) * self.pt_a + t2 * self.pt_b
        rect_side1 = contact_point_2 - self.contact_point_1
        if np.linalg.norm(rect_side1) < eps:
            # perpendicular direction is not defined
            raise GeometryNotFeasibleError()
        # TODO: handle case where denominator is zero
        t3_denom = np.dot(rect_side1, self.dc_vec)
        if t3_denom == 0:
            # Return the trivial rectangle in this degenerate case. See the comment in
            # `inscribed_area` about the geometry of this situation.
            return np.array(
                [
                    self.contact_point_1,
                    self.contact_point_1,
                    self.contact_point_1,
                    self.contact_point_1,
                ]
            )
        t3 = -(np.dot(rect_side1, self.cp_vec)) / t3_denom
        contact_point_3 = (1 - t3) * self.pt_c + t3 * self.pt_d
        rect_side2 = contact_point_3 - self.contact_point_1
        point4 = self.contact_point_1 + rect_side1 + rect_side2
        # return points in clockwise order (TODO test this)
        return np.array(
            [self.contact_point_1, contact_point_3, point4, contact_point_2]
        )

    def inscribed_area(self, t2, eps=1e-8):
        contact_point_2 = (1 - t2) * self.pt_a + t2 * self.pt_b
        rect_side1 = contact_point_2 - self.contact_point_1
        side1_norm2 = np.linalg.norm(rect_side1) ** 2
        if side1_norm2 < 1e-8:
            return 0.0

        # If (p2 - p1) is perpendicular to CD (`alpha_denom == 0` below), the third
        # contact point (and thus the inscribed area) is undetermined. For example, in
        # the unit square with pivot point p1 along the bottom edge,
        #
        # C-------B
        # |       |
        # |       |
        # |       |
        # D-------A (p2)
        # (p1)
        #
        # this edge case arises when t1 = 0.0 (so p1 == D), t2 = 0 (so p2 == A),
        # and so the normal to p2 - p1 is the entire segment CD. The maximum-area choice
        # would be to take the far vertex p3 == C as the third contact point. However,
        # we choose not to indulge this degenerate setup and so instead return an area
        # of zero (implicitly taking p3 == D, which is also valid) in this case.
        # This encourages the contact-point setup where p3 contacts along the top
        # edge rather than the left edge, which recovers the same solution without
        # the degenerate math.
        alpha_denom = np.dot(rect_side1, self.dc_vec)
        if alpha_denom == 0:
            return 0.0
        alpha = (self.K) / alpha_denom
        side2_norm2_alpha = alpha**2 * side1_norm2
        return np.sqrt(side1_norm2 * side2_norm2_alpha)

    def get_optimal_t2_and_area(self, valid_t2_intervals=None):
        if valid_t2_intervals is None:
            valid_t2_intervals = self.get_feasible_t2s()
        critical_points = self.critical_points_of_area_wrt_t2()

        candidate_t2s = []
        for t2_min, t2_max in valid_t2_intervals:
            candidate_t2s.append(t2_min)
            candidate_t2s.append(t2_max)
            for crit in critical_points:
                if crit > t2_min and crit < t2_max:
                    candidate_t2s.append(crit)

        if len(candidate_t2s) == 0:
            raise GeometryNotFeasibleError()

        areas = [self.inscribed_area(t2) for t2 in candidate_t2s]
        # ("candtdates", candidate_t2s)
        # print("areas", areas)
        best_t2_idx = np.nanargmax(areas)
        return candidate_t2s[best_t2_idx], areas[best_t2_idx]

    def plot(self, t2=None, ax=None):
        """Plots the geometry of this setup."""
        from matplotlib import pylab as plt

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        centroid = np.mean(self.quad, axis=0)
        radius = np.max(np.linalg.norm(self.quad - centroid, axis=-1))

        for edge in self.edges:
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], "k-")
        for label, pt in zip("ABCD", (self.pt_a, self.pt_b, self.pt_c, self.pt_d)):
            outwards = pt - centroid
            label_point = pt + 0.1 * outwards
            ax.text(label_point[0], label_point[1], label)

        if t2 == "optimize":
            t2, _ = self.get_optimal_t2_and_area()

        if t2 is not None:
            [p1, p3, p4, p2] = self.rect_from_t2(t2)
            print("plotting inscribed rect", [p1, p3, p4, p2])
            for rect_edge in [(p1, p2), (p1, p3), (p2, p4), (p3, p4)]:
                if np.linalg.norm(rect_edge[0] - centroid) >= 3 * radius:
                    continue
                if np.linalg.norm(rect_edge[1] - centroid) >= 3 * radius:
                    continue
                ax.plot(
                    [rect_edge[0][0], rect_edge[1][0]],
                    [rect_edge[0][1], rect_edge[1][1]],
                    "r--",
                )
            for label, pt in zip(("p1", "p2", "p3", "p4"), (p1, p2, p3, p4)):
                outwards = pt - centroid
                if np.linalg.norm(outwards) >= 3 * radius:
                    continue
                label_point = pt + 0.1 * outwards
                ax.text(label_point[0], label_point[1], label)
        else:
            p1 = self.contact_point_1
            outwards = p1 - centroid
            label_p1 = p1 + 0.1 * outwards
            ax.text(label_p1[0], label_p1[1], "p1")
        ax.grid(True)
        ax.axis("equal")  # Ensures equal scaling of axes
        # ax.set_xlim([centroid[0] -radius * 1.2, centroid[0] + radius * 1.2])
        # ax.set_ylim([centroid[1]-radius * 1.2, centroid[1] + radius * 1.2])
        plt.show()


def enumerate_three_contact_edges():
    """Enumerate all plausible triples of edges for contact points.

    There are 24 possibilities: twelve in which we have three unique edges,
    and twelve more in which two points lie on the same edge.

    Three unique edges: we have four choices for the initial edge containing
    the 'pivot' point p1. For each of these, there are (3 choose 2) = 3 possible
    choices for the remaining two contact edges (order doesn't matter).

    One doubled edge: again we have four choices for the edge
    that will contain both the 'pivot' point p1 and its adjecent contact point
    p2. For each of these, we choose one of the three remaining edges for
    the final contact point p3.
    """
    for i in range(4):
        for j in range(i, 4):
            for k in range(4):
                if (k == i) or k == j:
                    continue
                yield (i, j, k)


def largest_inscribed_rectangle(
    corner_points: BoundingBoxAny, num_eval_points=11
) -> tuple[QuadArray, float]:
    corner_points = bounding_box_as_array(corner_points)
    # Note that
    corner_points = geometry.sort_clockwise(corner_points)

    best_area = -1.0
    best_contacts = (None, None, None)
    best_t1 = None

    for i, j, k in enumerate_three_contact_edges():

        def negative_area_fn(t1):
            # print("trying with", i, j, k, t1)
            g = InscriptionGeometry(corner_points, i, j, k, t1=t1)
            try:
                _, area = g.get_optimal_t2_and_area()
            except GeometryNotFeasibleError:
                area = -1.0
            return -area

        r = minimize_scalar_multimodal(
            negative_area_fn, bounds=[0, 1], n_eval_points=num_eval_points
        )
        area = -r.fun
        if area > best_area:
            best_area = area
            best_contacts = (i, j, k)
            best_t1 = r.x

    best_i, best_j, best_k = best_contacts
    best_g = InscriptionGeometry(corner_points, best_i, best_j, best_k, t1=best_t1)
    # print("trying with best", best_i, best_j, best_k, best_t1)
    t2, _ = best_g.get_optimal_t2_and_area()
    best_rect = best_g.rect_from_t2(t2=t2)
    # print("best rect", best_rect)
    # print("best area", best_area)
    # print("best t2", t2)
    return best_rect, best_area
