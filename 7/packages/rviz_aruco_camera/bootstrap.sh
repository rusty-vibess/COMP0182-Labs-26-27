#!/usr/bin/env bash
set -eo pipefail

if [[ ${1:-} == --help ]]; then
  printf 'Usage: %s [workspace]\n' "${0##*/}"
  printf 'Source ROS Humble or Jazzy first. The default workspace is ~/camera_stream_ws.\n'
  exit 0
fi

if [[ $# -gt 1 ]]; then
  printf 'Usage: %s [workspace]\n' "${0##*/}" >&2
  exit 2
fi

case "${ROS_DISTRO:-}" in
  humble|jazzy) ;;
  *) printf 'Please source ROS Humble or Jazzy first.\n' >&2; exit 2 ;;
esac

package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace=${1:-"$HOME/camera_stream_ws"}
mkdir -p -- "$workspace"
cd -- "$workspace"

# match the install layout when using an existing workspace
layout=()
if [[ -f install/.colcon_install_layout && $(< install/.colcon_install_layout) == merged ]]; then
  layout=(--merge-install)
fi

# install dependencies and build just the camera display
rosdep install --from-paths "$package_dir" --ignore-src -y
colcon build --base-paths "$package_dir" --packages-select rviz_aruco_camera "${layout[@]}"

printf '\nBefore starting RViz, run:\n  source %q\n' "$PWD/install/setup.bash"
printf 'In RViz, add CameraStreamH264 and select your H.264 topic.\n'
