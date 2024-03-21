import pytest

from pyscope.telrun import plot_schedule_gantt, plot_schedule_sky, schedtel


def test_schedtel(tmp_path):
    catalog = "./tests/bin/test_schedtel.cat"
    observatory = "./tests/bin/simulator_observatory.cfg"

    schedule = schedtel(
        catalog=catalog,
        observatory=observatory,
        filename=str(tmp_path) + "test_schedtel.ecsv",
    )

    fig, ax = plot_schedule_gantt(schedule, observatory)
    fig.savefig(str(tmp_path) + "test_schedtel_gantt.png", bbox_inches="tight")

    fig, ax = plot_schedule_sky(schedule, observatory)
    fig.savefig(str(tmp_path) + "test_schedtel_sky.png", bbox_inches="tight")
