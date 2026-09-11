"""qe-report — self-contained HTML reports.

A report that shows only a total fails Phase 10: every score renders with its
constituents, and flags render beside a score rather than inside it.
"""

from .tearsheet import TearsheetInputs, render_tearsheet, sparkline_svg, summary_stats

__version__ = "0.1.0"

__all__ = ["TearsheetInputs", "render_tearsheet", "sparkline_svg", "summary_stats"]
