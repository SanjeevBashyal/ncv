#!/usr/bin/env bash
set -eu

pyuic5 -x ncv/ui/main_window.ui -o ncv/pyui/ui_main_window.py
pyuic5 -x ncv/ui/scatter_panel.ui -o ncv/pyui/ui_scatter_panel.py
pyuic5 -x ncv/ui/contour_panel.ui -o ncv/pyui/ui_contour_panel.py
pyuic5 -x ncv/ui/map_panel.ui -o ncv/pyui/ui_map_panel.py
pyuic5 -x ncv/ui/map_unavailable.ui -o ncv/pyui/ui_map_unavailable.py
