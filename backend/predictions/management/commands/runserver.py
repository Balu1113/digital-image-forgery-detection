"""``runserver`` override that also boots the Flask inference service.

Django discovers this command because ``predictions`` is in ``INSTALLED_APPS``
(listed before ``django.contrib.staticfiles``, whose own ``runserver`` would
otherwise win the precedence contest).  Behaviour is identical to the
staticfiles-enhanced command plus:

* the Flask app on ``settings.FORGERY_API_URL`` is started (if needed)
  before the HTTP server binds,
* it is stopped again when the dev server exits,
* ``--no-inference`` disables the auto-start for a single run.
"""

from django.contrib.staticfiles.management.commands.runserver import (
    Command as BaseRunserverCommand,
)

from predictions.inference_supervisor import InferenceSupervisor


class Command(BaseRunserverCommand):
    help = (
        "Starts a lightweight Web server for development. "
        "Also starts the Flask inference service unless --no-inference is given."
    )

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--no-inference",
            action="store_true",
            dest="no_inference",
            help="Do not auto-start the Flask inference service.",
        )

    def execute(self, *args, **options):
        supervisor = None
        if not options.get("no_inference"):
            supervisor = InferenceSupervisor(verbosity=options.get("verbosity", 1))
            supervisor.start()
        try:
            super().execute(*args, **options)
        finally:
            if supervisor is not None:
                supervisor.stop()
