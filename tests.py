import unittest
from pprint import pprint

import numpy

import pysupercluster


class SuperClusterTest(unittest.TestCase):
    def test_clustering1(self):
        points = numpy.array([
            (2.3522, 48.8566),   # 0: paris

            (-0.1278, 51.5074),  # 1: london
            (-0.0077, 51.4826),  # 2: greenwich
        ])

        index = pysupercluster.SuperCluster(
            points,
            min_zoom=0,
            max_zoom=16,
            radius=40,
            extent=512
        )

        clusters = index.getClusters(
            top_left=(-180, 90),
            bottom_right=(180, -90),
            zoom=4
        )

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
            zoom=0
        )

        self.assertEqual(len(clusters), 1)

        # cluster (all)
        self.assertEqual(clusters[0]['count'], 3)
        self.assertEqual(clusters[0]['child_ids'], [0, 1, 2])
        self.assertEqual(clusters[0]['expansion_zoom'], 3)
        self.assertEqual(clusters[0]['id'], 4)
        self.assertAlmostEqual(clusters[0]['latitude'], 50.6317143)
        self.assertAlmostEqual(clusters[0]['longitude'], 0.7389)

    def test_clustering_with_duplicate(self):
        points = numpy.array([
            (2.3522, 48.8566),   # 0: paris

            (-0.1278, 51.5074),  # 1: london
            (-0.1278, 51.5074),  # 2: london (again)
            (-0.0077, 51.4826),  # 3: greenwich
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
        self.assertEqual(clusters[1]['count'], 3)
        self.assertEqual(clusters[1]['child_ids'], [1, 2, 3])
        self.assertEqual(clusters[1]['expansion_zoom'], 8)
        self.assertEqual(clusters[1]['id'], 5)
        self.assertAlmostEqual(clusters[1]['latitude'], 51.49913483250177)
        self.assertAlmostEqual(clusters[1]['longitude'], -0.08776666666666211)

    def test_clustering_bug1(self):
        points = numpy.array([
            (-115.6825, 36.3091667),  # 0: MCWILLIAMS
            (-115.6694, 36.3086111),  # 1: Foxtail Group Picnic Area

            (-115.4192, 36.1462660),  # 2: Red Spring Picnic Area

            (-115.4563, 36.0710988),  # 3: Group Day Use Areas
            (-115.4563, 36.0749881),  # 4: Shelters
            (-115.4460, 36.0739119),  # 5: Horseshoe Horse Stalls
        ])

        index = pysupercluster.SuperCluster(
            points,
            min_zoom=0,
            max_zoom=12,
            radius=52,
            extent=512
        )

        clusters = index.getClusters(
            top_left=(-116.25561908694154, 36.4522124053111),
            bottom_right=(-114.88934475400278, 35.85740973830082),
            zoom=11
        )

        # pprint(clusters)

        self.assertEqual(len(clusters), 3)

        # cluster (0, 1)
        self.assertEqual(clusters[0]['count'], 2)
        self.assertEqual(clusters[0]['child_ids'], [0, 1])
        self.assertEqual(clusters[0]['expansion_zoom'], 12)
        self.assertEqual(clusters[0]['id'], 7)
        self.assertAlmostEqual(clusters[0]['latitude'], 36.3088889)
        self.assertAlmostEqual(clusters[0]['longitude'], -115.67595)

        # single point (2)
        self.assertEqual(clusters[1]['count'], 1)
        self.assertEqual(clusters[1]['child_ids'], [])
        self.assertIsNone(clusters[1]['expansion_zoom'])
        self.assertEqual(clusters[1]['id'], 2)
        self.assertAlmostEqual(clusters[1]['latitude'], 36.1462660)
        self.assertAlmostEqual(clusters[1]['longitude'], -115.4192)

        # cluster (3, 4, 5)
        self.assertEqual(clusters[2]['count'], 3)
        self.assertEqual(clusters[2]['child_ids'], [3, 4, 5])
        self.assertEqual(clusters[2]['expansion_zoom'], 12)
        self.assertEqual(clusters[2]['id'], 8)
        self.assertAlmostEqual(clusters[2]['latitude'], 36.07333295042622)
        self.assertAlmostEqual(clusters[2]['longitude'], -115.45286666666667)

    def test_clustering_bug2(self):
        points = numpy.array([
            (-115.3838889, 36.1313889),    # 0: Red Rock Canyon Campground

            (-115.419201, 36.146266),      # 1: Red Spring Picnic Area

            (-115.45635166, 36.07109884),  # 2: Group Day Use Areas
            (-115.45635166, 36.07109884),  # 3: Shelters (dup)
            (-115.1357397, 36.1806652),    # 4: Horse Stalls
            (-115.4460328, 36.0739119),    # 5: Horseshoe Horse Stalls
            (-115.1357397, 36.1806652),    # 6: Backpack Parking (dup: 4)
            (-115.1357397, 36.1806652),    # 7: Hunts 1 (dup: 4)
            (-115.4460328, 36.0739119),    # 8: Hunts 2
            (-115.45635166, 36.07109884),  # 9: Paddle In Campground

            (0, 0),                        # 10: Las Vegas Bay Campground (off-map)
        ])

        index = pysupercluster.SuperCluster(
            points,
            min_zoom=0,
            max_zoom=12,
            radius=52,
            extent=512
        )

        clusters = index.getClusters(
            top_left=(-116.19121334347108, 36.23679057416504),
            bottom_right=(-114.850692900978, 35.65163615086732),
            zoom=11
        )

        self.assertEqual(len(clusters), 4)  # 2 clusters, 2 points

        # single point (0)
        self.assertEqual(clusters[0]['count'], 1)
        self.assertEqual(clusters[0]['child_ids'], [])
        self.assertIsNone(clusters[1]['expansion_zoom'])
        self.assertEqual(clusters[0]['id'], 0)
        self.assertAlmostEqual(clusters[0]['latitude'], 36.1313889)
        self.assertAlmostEqual(clusters[0]['longitude'], -115.3838889)

        # single point (1)
        self.assertEqual(clusters[1]['count'], 1)
        self.assertEqual(clusters[1]['child_ids'], [])
        self.assertIsNone(clusters[1]['expansion_zoom'])
        self.assertEqual(clusters[1]['id'], 1)
        self.assertAlmostEqual(clusters[1]['latitude'], 36.14626600000001)
        self.assertAlmostEqual(clusters[1]['longitude'], -115.419201)

        # cluster (?)
        self.assertEqual(clusters[2]['count'], 5)
        self.assertEqual(clusters[2]['child_ids'], [2, 3, 5, 8, 9])
        self.assertEqual(clusters[2]['expansion_zoom'], 12)
        self.assertEqual(clusters[2]['id'], 14)
        # self.assertAlmostEqual(clusters[2]['latitude'], 36.3088889)
        # self.assertAlmostEqual(clusters[2]['longitude'], -115.67595)

        # cluster (4, 6, 7)
        self.assertEqual(clusters[3]['count'], 3)
        self.assertEqual(clusters[3]['child_ids'], [4, 6, 7])
        self.assertEqual(clusters[3]['expansion_zoom'], 13)
        self.assertEqual(clusters[3]['id'], 12)
        self.assertAlmostEqual(clusters[3]['latitude'], 36.18066519999999)
        self.assertAlmostEqual(clusters[3]['longitude'], -115.1357397)

    def test_clustering_bug3(self):
        points = numpy.array([
            (-115.6694444, 36.3086111),    # 0: FOXTAIL GRP PICNIC AREA
            (-115.5853333, 36.2722222),    # 1: SPRING MOUNTAINS VISITOR GATEWAY GROUP PICNIC SITES
            (-115.6161111, 36.3119444),    # 2: MAHOGANY GROVE
            (-115.3838889, 36.1313889),    # 3: Red Rock Canyon Campground
            (-115.6066639, 36.3098639),    # 4: HILLTOP
            (-115.6144444, 36.2633333),    # 5: FLETCHER VIEW
            (-115.419201, 36.146266),      # 6: Red Spring Picnic Area
            (-115.6075, 36.2625),          # 7: KYLE CANYON PICNIC AREA DAY USE
            (-115.6438889, 36.2566667),    # 8: CATHEDRAL ROCK PICNIC AREA
            (-115.6825, 36.3091667),       # 9: MCWILLIAMS
            (-115.45635166, 36.07109884),  # 10: Group Day Use Areas
            (-115.45635166, 36.07109884),  # 11: Shelters
            (-115.1357397, 36.1806652),    # 12: Horse Stalls
            (-115.4460328, 36.0739119),    # 13: Horseshoe Horse Stalls
            (-115.1357397, 36.1806652),    # 14: Backpack Parking
            (-115.1357397, 36.1806652),    # 15: Hunts
            (-115.4460328, 36.0739119),    # 16: Hunts
            (-115.45635166, 36.07109884),  # 17: Paddle In Campground
            (114.4807443, 36.0207625),     # 18: Boulder Beach Campground
            (0, 0),                        # 19: Las Vegas Bay Campground
            (0, 0),                        # 20: Boulder Beach Group Campsites
            (40.613599, -57.397734),       # 21: Whiskeytown Gold Panning Pass
            (-109.919942, 38.681923),      # 22: Dellenbaugh Tunnel Trailhead
            (-109.702327, 40.572019),      # 23: Dry Fork Flume Trailhead #1
            (0, 0),                         # 24: Callville Bay Campground
            (-85.24262015365, 30.819760948723),  # 25: Loop BH1
        ])

        visited_child_ids = set()

        index = pysupercluster.SuperCluster(
            points,
            min_zoom=0,
            max_zoom=12,
            radius=52,
            extent=512
        )

        clusters = index.getClusters(
            top_left=(-116.33593974517714, 36.36433946898194),
            bottom_right=(-114.62981903577308, 35.69693608381462),
            zoom=11
        )

        pprint(clusters)

        # self.assertEqual(len(clusters), ?)

        for cluster in clusters:
            self.assertIsNotNone(cluster['id'])
            self.assertIsNotNone(cluster['latitude'])
            self.assertIsNotNone(cluster['longitude'])

            count = cluster['count']
            self.assertIsNotNone(count)
            self.assertGreater(count, 0)

            child_ids = cluster['child_ids']
            self.assertIsInstance(child_ids, list)

            for child_id in child_ids:
                self.assertFalse(child_id in visited_child_ids, f"child_id {child_id} appears in >1 cluster")
                visited_child_ids.add(child_id)

            if count > 1 or len(child_ids) > 0:
                # Cluster
                self.assertEqual(len(child_ids), count)
                self.assertIsNotNone(cluster['expansion_zoom'])
                self.assertGreaterEqual(cluster['id'], len(points))
            else:
                # Point
                self.assertIsNone(cluster['expansion_zoom'])
                self.assertLess(cluster['id'], len(points))

                point_lon, point_lat = points[cluster['id']]
                self.assertAlmostEqual(point_lon, cluster['longitude'])
                self.assertAlmostEqual(point_lat, cluster['latitude'])

    def test_empty_input(self):
        points = numpy.ones((0, 2))

        with self.assertRaises(ValueError):
            index = pysupercluster.SuperCluster(
                points,
                min_zoom=0,
                max_zoom=16,
                radius=40,
                extent=512
            )


if __name__ == '__main__':
    unittest.main()
