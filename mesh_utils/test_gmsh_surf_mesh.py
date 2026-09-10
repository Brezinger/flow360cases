import unittest
from unittest.mock import patch

import gmsh_surf_mesh


class DefaultGmshThreadCountTests(unittest.TestCase):
    @patch("gmsh_surf_mesh.psutil.cpu_count", return_value=16)
    def test_uses_one_less_than_physical_core_count(self, _cpu_count) -> None:
        self.assertEqual(gmsh_surf_mesh.default_gmsh_thread_count(), 15)

    @patch("gmsh_surf_mesh.psutil.cpu_count", return_value=1)
    def test_keeps_at_least_one_thread(self, _cpu_count) -> None:
        self.assertEqual(gmsh_surf_mesh.default_gmsh_thread_count(), 1)

    @patch("gmsh_surf_mesh.os.cpu_count", return_value=8)
    @patch("gmsh_surf_mesh.psutil", None)
    def test_falls_back_to_logical_cpu_count(self, _cpu_count) -> None:
        self.assertEqual(gmsh_surf_mesh.default_gmsh_thread_count(), 7)


if __name__ == "__main__":
    unittest.main()
