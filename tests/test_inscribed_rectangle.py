import numpy as np

from core import geometry, inscribed_rectangle

eps = 1e-4


def assert_matching_rect(r1, r2, atol=0.0):
    # Check that each point in r1 occurs in r2.
    r1, r2 = np.asarray(r1), np.asarray(r2)
    assert len(r1) == len(r2)
    for pt in r1:
        dists = np.linalg.norm(pt - r2, axis=-1)
        assert np.min(dists) < atol


def assert_rect_is_inscribed(quad_vertices, rect_vertices, atol=0.0):
    # Check that all rectangle corners lie inside the bounding quadrilateral.
    a, b, c, d = quad_vertices
    for pt in rect_vertices:
        for v1, v2 in [(a, b), (b, c), (c, d), (d, a)]:
            cond = inscribed_rectangle.cross2d(v2 - v1, pt - v1)
            # print(f"pt {pt} edge {v2} {v1} cond {cond}")
            assert cond >= -atol


class TestInscriptionGeometry:
    def test_feasible_unit_square_offset(self):
        square_vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        g = inscribed_rectangle.InscriptionGeometry(square_vertices, 0, 1, 3, t1=0.2)

        cutpoints_t3 = g.get_t2_cutpoints_from_third_edge_bounds()
        assert len(cutpoints_t3) == 1
        assert abs(cutpoints_t3[0] - 0.16) < eps

        cutpoints_p4 = sorted(g.get_t2_cutpoints_from_edge_halfplane_constraints())
        assert len(cutpoints_p4) == 4
        expected_cutpoints = np.array([0.0, 0.0, 0.2, 0.8])
        cutpoints_p4 = np.array(cutpoints_p4)
        assert np.max(np.abs(cutpoints_p4 - expected_cutpoints)) < eps

        feasible_t2s = g.get_feasible_t2s()
        assert len(feasible_t2s) == 1
        feasible_interval = np.array(feasible_t2s[0])
        expected_feasible_interval = np.array([0.2, 0.8])
        assert np.max(np.abs(feasible_interval - expected_feasible_interval)) < eps

        t2, area = g.get_optimal_t2_and_area(feasible_t2s)
        assert abs(t2 - 0.2) < eps
        assert abs(area - 0.68) < eps

    def test_unit_square_aligned(self):
        square_vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        g = inscribed_rectangle.InscriptionGeometry(square_vertices, 0, 1, 2, t1=0.0)
        t2, area = g.get_optimal_t2_and_area()
        assert abs(t2 - 0.0) < eps
        assert abs(area - 1.0) < eps

    def test_zero_width_inscribed_rectangle(self):
        square_vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        t1 = 0.9
        g = inscribed_rectangle.InscriptionGeometry(square_vertices, 3, 3, 1, t1=t1)
        area = g.inscribed_area(t2=t1)
        assert area == 0.0

        raised_not_feasible = False
        try:
            _ = g.rect_from_t2(t2=t1)
        except inscribed_rectangle.GeometryNotFeasibleError:
            raised_not_feasible = True
        assert raised_not_feasible

    def test_includes_linear_halfplane_constraints(self):
        # Test parameters in which p4 ends up being constrained by the same
        # edge AB that p2 lies on, which causes the quadratic constraint equation
        # to become linear. Not handling this will either lead to nans or to
        # missing the constraint on p4 (and thus generating an invalid inscribed
        # rectangle).
        quad_vertices = np.array(
            [
                [23.93967607, 790.97519764],
                [835.0071736, 774.74892396],
                [857.96269727, 1357.24533716],
                [40.17216643, 1388.80918221],
            ]
        )
        g = inscribed_rectangle.InscriptionGeometry(
            quad_vertices, 2, 3, 1, t1=0.0005783888773943481
        )
        # Check that our t2 cutpoints include the one from edge AB
        halfplane_cuts = g.get_t2_cutpoints_from_edge_halfplane_constraints()
        assert min([abs(x - 0.015634362175197737) for x in halfplane_cuts]) < eps

        # Check that the rectangle is fully inscribed.
        t2, _ = g.get_optimal_t2_and_area()
        rect = g.rect_from_t2(t2)
        assert_rect_is_inscribed(quad_vertices, rect, atol=eps)


class TestLargestInscribedRectangle:
    def test_unit_square(self):
        square_vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        rect, area = inscribed_rectangle.largest_inscribed_rectangle(square_vertices)
        assert_rect_is_inscribed(square_vertices, rect, atol=eps)
        assert abs(area - 1.0) < eps
        assert_matching_rect(rect, square_vertices, atol=eps)

    def test_diamond(self):
        diamond_vertices = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        rect, area = inscribed_rectangle.largest_inscribed_rectangle(diamond_vertices)
        assert_rect_is_inscribed(diamond_vertices, rect, atol=eps)
        assert abs(area - 2.0) < eps
        assert_matching_rect(rect, diamond_vertices, atol=eps)

    def test_skinny_trapezoid(self):
        trap_vertices = np.array([(-10.0, 0.0), (10.0, 0.0), (1.0, 10.0), (-1.0, 10.0)])
        rect, area = inscribed_rectangle.largest_inscribed_rectangle(trap_vertices)
        assert_rect_is_inscribed(trap_vertices, rect, atol=eps)
        assert abs(area - 55.555555555555) < eps
        assert_matching_rect(
            rect,
            np.array([[5.0, 0.0], [5.0, 5.55555556], [-5.0, 5.55555556], [-5.0, 0.0]]),
            atol=eps,
        )

    def test_irregular_quadrilateral(self):
        quad_vertices = geometry.sort_clockwise(
            np.array([(-5.0, -5.0), (-4.0, 5.0), (5.0, 4.0), (2.0, -4.0)])
        )
        rect, area = inscribed_rectangle.largest_inscribed_rectangle(quad_vertices)
        assert_rect_is_inscribed(quad_vertices, rect, atol=eps)
        assert abs(area - 56.86809386891749) < eps
        assert_matching_rect(
            rect,
            np.array(
                [
                    [2.82417122, 4.24175875],
                    [2.0, -4.0],
                    [-4.83167894, -3.31683596],
                    [-4.00750772, 4.92492279],
                ]
            ),
            atol=eps,
        )
