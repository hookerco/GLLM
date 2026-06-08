import unittest

import matplotlib

matplotlib.use("Agg")

from gllm.utils.plot_utils import plot_user_specification


class PlotUtilsTests(unittest.TestCase):
    def test_user_spec_plot_limits_include_negative_tool_path_points(self):
        plot_module = plot_user_specification(
            {
                "workpiece_diemensions": [12.0, 12.0],
                "starting_point": [0.0, 0.0, 0.1],
                "tool_path": [
                    (3.0, 0.0, 0.0),
                    (0.0, 3.0, 0.0),
                    (-3.0, 0.0, 0.0),
                    (0.0, -3.0, 0.0),
                    (3.0, 0.0, 0.0),
                ],
                "cut_depth": [0.25],
            }
        )

        ax = plot_module.gca()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        self.assertLessEqual(x_min, -3.0)
        self.assertGreaterEqual(x_max, 12.0)
        self.assertLessEqual(y_min, -3.0)
        self.assertGreaterEqual(y_max, 12.0)
        plot_module.close()


if __name__ == "__main__":
    unittest.main()
