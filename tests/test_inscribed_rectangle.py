import numpy as np

import pytest

from image_processing import inscribed_rectangle

eps = 1e-4

def assert_matching_rect(r1, r2, atol=0.):
    # Check that each point in r1 occurs in r2.
    r1, r2 = np.asarray(r1), np.asarray(r2)
    assert len(r1) == len(r2)
    for pt in r1:
        dists = np.linalg.norm(pt - r2, axis=-1)
        assert np.min(dists) < atol

class TestInscriptionGeometry:
    

    def test_feasible_unit_square_offset(self):
        square_vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        g = inscribed_rectangle.InscriptionGeometry(
            square_vertices, 0, 1, 3, t1=0.2)
        
        cutpoints_t3 = g.get_t2_cutpoints_from_third_edge_bounds()
        assert len(cutpoints_t3) == 1
        assert abs(cutpoints_t3[0] - 0.16) < eps
        
        cutpoints_p4 = sorted(
            g.get_t2_cutpoints_from_edge_halfplane_constraints())
        assert len(cutpoints_p4) == 2
        expected_cutpoints = np.array([0.2, 0.8])
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
        g = inscribed_rectangle.InscriptionGeometry(
            square_vertices, 0, 1, 2, t1=0.0)
        t2, area = g.get_optimal_t2_and_area()
        assert abs(t2 - 0.0) < eps
        assert abs(area - 1.0) < eps
        
    def test_zero_width_inscribed_rectangle(self):
        square_vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        t1 = 0.9
        g = inscribed_rectangle.InscriptionGeometry(
            square_vertices, 3, 3, 1, t1=t1)
        area = g.inscribed_area(t2=t1)
        assert area == 0.
        
        raised_not_feasible = False
        try:
            rect = g.rect_from_t2(t2=t1)
        except inscribed_rectangle.NotFeasibleException:
            raised_not_feasible = True
        assert raised_not_feasible

class TestLargestInscribedRectangle:
    
    def test_unit_square(self):
        square_vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        rect, area = inscribed_rectangle.largest_inscribed_rectangle(square_vertices)
        assert abs(area - 1.0) < eps
        assert_matching_rect(rect, square_vertices, atol=eps)
        
    def test_diamond(self):
        diamond_vertices = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        rect, area = inscribed_rectangle.largest_inscribed_rectangle(diamond_vertices)
        assert abs(area - 2.0) < eps
        assert_matching_rect(rect, diamond_vertices, atol=eps)

    def test_skinny_trapezoid(self):
        trap_vertices = np.array([(-10., 0.), (10., 0.), (1., 10.), (-1., 10.)])
        rect, area = inscribed_rectangle.largest_inscribed_rectangle(trap_vertices)
        assert abs(area - 55.555555555555) < eps
        assert_matching_rect(
            rect, 
            np.array([[ 5.,          0.        ],
                      [ 5.,          5.55555556],
                      [-5.,          5.55555556],
                      [-5.,          0.        ]]), 
            atol=eps)
        
    def test_irregular_quadrilateral(self):
        quad_vertices = np.array([(-5., -5.), (-4., 5.), (5., 4.), (2., -4.)])
        rect, area = inscribed_rectangle.largest_inscribed_rectangle(quad_vertices)
        assert abs(area - 57.62184976310794) < eps
        assert_matching_rect(
            rect,
            np.array([[ 2.91462208,  4.23170866],
                      [ 2.,         -4.        ],
                      [-4.9146231,  -3.2317188 ],
                      [-4.00000101,  4.99998985]]),
            atol=eps)
        
        
        