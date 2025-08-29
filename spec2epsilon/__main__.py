# spec2epsilon/__main__.py

import sys
import os
import streamlit.web.cli as stcli

def main():
    """Entry point for the `spec2epsilon` console command."""
    # Resolve the path to your Streamlit app script
    app_path = os.path.join(os.path.dirname(__file__), "app.py")

    # Equivalent to running: streamlit run app.py
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()

