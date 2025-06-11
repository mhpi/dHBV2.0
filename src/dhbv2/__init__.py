import logging
import os
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from platformdirs import user_config_dir

log = logging.getLogger('dhbv2')

__all__ = [
    '__version__',
    'available_models',
    'available_modules',
    'load_model',
    'load_module',
]


def _check_license_agreement():
    """Checks if user has agreed to package license and prompts if not."""
    package_name = 'dhbv2'

    config_dir = Path(user_config_dir(package_name))
    agreement_file = config_dir / '.license_status'

    if not agreement_file.exists():
        print(f"\n[----- {package_name} LICENSE AGREEMENT -----]")

        try:
            # Find and read LICENSE file
            license_path = files(package_name).parent.parent.joinpath("LICENSE")
            license = license_path.read_text(encoding="utf-8")
            print(license)
        except FileNotFoundError:
            # Fallback in case the LICENSE file wasn't packaged correctly
            print(
                "\n|> Error locating License. Showing summary <|\n" \
                "By using this software, you agree to the terms specified \n" \
                "in the Non-Commercial Software License Agreement: \n" \
                "\nhttps://github.com/mhpi/dhbv2/blob/master/LICENSE \n" \
                "\n'dhbv2' is free for non-commercial use. \n" \
                "Prior authorization must be obtained for commercial \n" \
                "use. For further details, please contact the Pennsylvania \n" \
                "State University Office of Technology Management at \n" \
                "814.865.6277 or otminfo@psu.edu.\n",
            )

        print("-" * 40)

        response = input("Do you agree to these terms? Type 'Yes' to continue: ")
        
        if response.strip().lower() in ['yes', 'y']:
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
                agreement_file.write_text(
                    f"accepted_on = {datetime.now().isoformat()}Z\nversion = 1\n",
                    encoding="utf-8",
                )
                log.warning(f"License accepted. Agreement written to {agreement_file}\n")
            except OSError as e:
                log.warning(
                    f"Failed to save agreement file {agreement_file}: {e}")
                print(
                    "You may need to run with administrator privileges to avoid " \
                    "repeating this process at runtime.",
                )
        else:
            print("\n>| License agreement not accepted. Exiting. <|")
            raise SystemExit(1)


# This only runs once when package is first imported.
if not os.environ.get('CI'):
    # Skip license check in CI envs (e.g., GitHub Actions)
    _check_license_agreement()
