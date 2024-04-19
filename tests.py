import unittest

import numpy

import pysupercluster


class SuperClusterTest(unittest.TestCase):
    def test_clustering(self):
        points = numpy.array([
            (2.3522, 48.8566),   # paris
            (-0.1278, 51.5074),  # london
            (-0.0077, 51.4826),  # greenwich
        ])

        index = pysupercluster.SuperCluster(
            points,
            min_zoom=0,
            max_zoom=16,
            radius=40,
            extent=512)

        clusters = index.getClusters(
            top_left=(-180, 90),
            bottom_right=(180, -90),
            zoom=4)

        self.assertEqual(len(clusters), 2)

        # single point (paris)
        self.assertEqual(clusters[0]['count'], 1)
        self.assertEqual(clusters[0]['child_ids'], [])
        self.assertIsNone(clusters[0]['expansion_zoom'])
        self.assertEqual(clusters[0]['id'], 0)
        self.assertAlmostEqual(clusters[0]['latitude'], 48.8566)
        self.assertAlmostEqual(clusters[0]['longitude'], 2.3522)

        # cluster (london, greenwich)
        self.assertEqual(clusters[1]['count'], 2)
        self.assertEqual(clusters[1]['child_ids'], [1, 2])
        self.assertEqual(clusters[1]['expansion_zoom'], 8)
        self.assertEqual(clusters[1]['id'], 3)
        self.assertAlmostEqual(clusters[1]['latitude'], 51.4950017)
        self.assertAlmostEqual(clusters[1]['longitude'], -0.0677500)

        clusters = index.getClusters(
            top_left=(-180, 90),
            bottom_right=(180, -90),
            zoom=0)

        self.assertEqual(len(clusters), 1)

        # cluster (all)
        self.assertEqual(clusters[0]['count'], 3)
        self.assertEqual(clusters[0]['child_ids'], [0, 1, 2])
        self.assertEqual(clusters[0]['expansion_zoom'], 3)
        self.assertEqual(clusters[0]['id'], 4)
        self.assertAlmostEqual(clusters[0]['latitude'], 50.6317143)
        self.assertAlmostEqual(clusters[0]['longitude'], 0.7389)

    def test_empty_input(self):
        points = numpy.ones((0, 2))

        with self.assertRaises(ValueError):
            index = pysupercluster.SuperCluster(
                points,
                min_zoom=0,
                max_zoom=16,
                radius=40,
                extent=512)


if __name__ == '__main__':
    unittest.main()
