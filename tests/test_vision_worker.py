
from __future__ import annotations

import json
import time

from threading import Event

from PIL import Image

from aura.perception.vision_types import (
    VisionResultStatus,
)

from aura.perception.vision_worker import (
    AsyncVisionWorker,
    VisionWorkerConfig,
)


def make_image():

    return Image.new(
        "RGB",
        (320, 180),
        color=(50, 80, 120),
    )


def submit(
    worker,
    frame_id,
    continuity_id=1,
    captured_at=None,
    image=None,
    context=None,
):

    return worker.submit(
        image=(
            image
            if image is not None
            else make_image()
        ),
        frame_id=frame_id,
        continuity_id=continuity_id,
        captured_at=(
            captured_at
            if captured_at is not None
            else time.monotonic()
        ),
        reason="synthetic_test",
        context=context,
    )


def simple_analyser(request):

    return {
        "frame_id": request.frame_id,
        "scene": "synthetic_gameplay",
        "image_size": list(request.image.size),
    }


def test_basic_inference():

    worker = AsyncVisionWorker(
        simple_analyser
    )

    worker.start()

    try:

        request_id = submit(worker, 1)

        assert worker.wait_until_idle(2.0)

        results = worker.poll_results()

        assert len(results) == 1

        result = results[0]

        assert result.request_id == request_id

        assert (
            result.status
            == VisionResultStatus.SUCCESS
        )

        assert result.analysis["frame_id"] == 1

        assert result.inference_ms >= 0

        json.dumps(
            result.to_dict(),
            allow_nan=False,
        )

    finally:

        assert worker.stop()

    print("PASS: Basic asynchronous inference")


def test_non_blocking_submission():

    entered = Event()
    release = Event()

    def slow_analyser(request):

        entered.set()

        if not release.wait(3.0):

            raise TimeoutError("Test gate timed out.")

        return simple_analyser(request)

    worker = AsyncVisionWorker(slow_analyser)

    worker.start()

    try:

        submit(worker, 1)

        assert entered.wait(2.0)

        start = time.perf_counter()

        submit(worker, 2)

        submission_ms = (
            time.perf_counter() - start
        ) * 1000

        assert submission_ms < 100.0

        status = worker.get_status()

        assert status["busy"]

        assert status["pending_requests"] == 1

        release.set()

        assert worker.wait_until_idle(3.0)

    finally:

        release.set()

        assert worker.stop()

    print(
        "PASS: Non-blocking submission "
        f"({submission_ms:.3f} ms)"
    )


def test_latest_frame_wins():

    entered = Event()
    release = Event()

    processed = []

    def analyser(request):

        processed.append(request.frame_id)

        if request.frame_id == 1:

            entered.set()

            if not release.wait(3.0):

                raise TimeoutError(
                    "Test gate timed out."
                )

        return simple_analyser(request)

    worker = AsyncVisionWorker(analyser)

    worker.start()

    try:

        submit(worker, 1)

        assert entered.wait(2.0)

        for frame_id in range(2, 11):

            submit(worker, frame_id)

            assert (
                worker.get_status()["pending_requests"]
                <= 1
            )

        release.set()

        assert worker.wait_until_idle(3.0)

        results = worker.poll_results()

        assert processed == [1, 10]

        assert [
            result.frame_id
            for result in results
        ] == [1, 10]

        stats = worker.get_status()["statistics"]

        assert stats["submitted"] == 10

        assert stats["started"] == 2

        assert stats["replaced"] == 8

    finally:

        release.set()

        assert worker.stop()

    print("PASS: Latest-frame-wins queue")


def test_context_invalidation():

    entered = Event()
    release = Event()

    def analyser(request):

        if request.frame_id == 1:

            entered.set()

            if not release.wait(3.0):

                raise TimeoutError(
                    "Test gate timed out."
                )

        return simple_analyser(request)

    worker = AsyncVisionWorker(analyser)

    worker.start()

    try:

        submit(worker, 1, continuity_id=1)

        assert entered.wait(2.0)

        submit(worker, 2, continuity_id=1)

        worker.invalidate(continuity_id=2)

        submit(worker, 3, continuity_id=2)

        release.set()

        assert worker.wait_until_idle(3.0)

        results = worker.poll_results()

        assert len(results) == 1

        assert results[0].frame_id == 3

        assert results[0].continuity_id == 2

        stats = worker.get_status()["statistics"]

        assert stats["invalidated_pending"] == 1

        assert stats["invalidated_in_flight"] == 1

    finally:

        release.set()

        assert worker.stop()

    print("PASS: In-flight context invalidation")


def test_automatic_continuity_change():

    worker = AsyncVisionWorker(
        simple_analyser
    )

    worker.start()

    try:

        submit(worker, 1, continuity_id=1)

        assert worker.wait_until_idle(2.0)

        submit(worker, 2, continuity_id=2)

        assert worker.wait_until_idle(2.0)

        results = worker.poll_results()

        assert len(results) == 1

        assert results[0].continuity_id == 2

        assert results[0].generation == 1

    finally:

        assert worker.stop()

    print("PASS: Automatic continuity invalidation")


def test_expired_request():

    worker = AsyncVisionWorker(
        simple_analyser,
        VisionWorkerConfig(
            maximum_request_age_seconds=0.5,
            maximum_result_age_seconds=2.0,
        ),
    )

    worker.start()

    try:

        submit(
            worker,
            1,
            captured_at=time.monotonic() - 1.0,
        )

        assert worker.wait_until_idle(2.0)

        assert not worker.poll_results()

        stats = worker.get_status()["statistics"]

        assert stats["started"] == 0

        assert stats["expired_before_start"] == 1

    finally:

        assert worker.stop()

    print("PASS: Expired request rejection")


def test_expired_result():

    entered = Event()
    release = Event()

    def analyser(request):

        entered.set()

        if not release.wait(3.0):

            raise TimeoutError("Test gate timed out.")

        return simple_analyser(request)

    worker = AsyncVisionWorker(
        analyser,
        VisionWorkerConfig(
            maximum_request_age_seconds=0.2,
            maximum_result_age_seconds=0.3,
        ),
    )

    worker.start()

    try:

        submit(worker, 1)

        assert entered.wait(1.0)

        time.sleep(0.4)

        release.set()

        assert worker.wait_until_idle(2.0)

        assert not worker.poll_results()

        stats = worker.get_status()["statistics"]

        assert stats["expired_after_inference"] == 1

    finally:

        release.set()

        assert worker.stop()

    print("PASS: Expired inference result rejection")


def test_error_recovery():

    def analyser(request):

        if request.frame_id == 1:

            raise RuntimeError(
                "Simulated inference failure"
            )

        return simple_analyser(request)

    worker = AsyncVisionWorker(analyser)

    worker.start()

    try:

        submit(worker, 1)

        assert worker.wait_until_idle(2.0)

        result = worker.poll_results()[0]

        assert (
            result.status
            == VisionResultStatus.ERROR
        )

        assert "Simulated" in result.error

        submit(worker, 2)

        assert worker.wait_until_idle(2.0)

        result = worker.poll_results()[0]

        assert (
            result.status
            == VisionResultStatus.SUCCESS
        )

        stats = worker.get_status()["statistics"]

        assert stats["failed"] == 1

        assert stats["succeeded"] == 1

    finally:

        assert worker.stop()

    print("PASS: Inference failure recovery")


def test_invalid_output():

    def analyser(request):

        return {
            "invalid_number": float("nan")
        }

    worker = AsyncVisionWorker(analyser)

    worker.start()

    try:

        submit(worker, 1)

        assert worker.wait_until_idle(2.0)

        result = worker.poll_results()[0]

        assert (
            result.status
            == VisionResultStatus.ERROR
        )

        assert result.analysis is None

    finally:

        assert worker.stop()

    print("PASS: Invalid model output rejection")


def test_snapshot_isolation():

    entered = Event()
    release = Event()

    def analyser(request):

        entered.set()

        if not release.wait(3.0):

            raise TimeoutError("Test gate timed out.")

        return {
            "pixel": list(
                request.image.getpixel((0, 0))
            ),
            "health": request.context["health"],
        }

    worker = AsyncVisionWorker(analyser)

    worker.start()

    image = make_image()

    context = {
        "health": 75.0
    }

    try:

        submit(
            worker,
            1,
            image=image,
            context=context,
        )

        assert entered.wait(2.0)

        image.putpixel(
            (0, 0),
            (255, 255, 255),
        )

        context["health"] = 10.0

        release.set()

        assert worker.wait_until_idle(2.0)

        result = worker.poll_results()[0]

        assert result.analysis["pixel"] == [
            50, 80, 120
        ]

        assert result.analysis["health"] == 75.0

    finally:

        release.set()

        assert worker.stop()

    print("PASS: Image and context snapshot isolation")


def test_bounded_result_buffer():

    worker = AsyncVisionWorker(
        simple_analyser,
        VisionWorkerConfig(
            maximum_buffered_results=3,
        ),
    )

    worker.start()

    try:

        for frame_id in range(1, 7):

            submit(worker, frame_id)

            assert worker.wait_until_idle(2.0)

        status = worker.get_status()

        assert status["buffered_results"] == 3

        results = worker.poll_results()

        assert [
            result.frame_id
            for result in results
        ] == [4, 5, 6]

        assert (
            status["statistics"][
                "result_buffer_overflows"
            ] == 3
        )

    finally:

        assert worker.stop()

    print("PASS: Bounded completed-result buffer")


def test_expired_on_poll():

    worker = AsyncVisionWorker(
        simple_analyser,
        VisionWorkerConfig(
            maximum_request_age_seconds=0.3,
            maximum_result_age_seconds=0.5,
        ),
    )

    worker.start()

    try:

        submit(worker, 1)

        assert worker.wait_until_idle(2.0)

        time.sleep(0.6)

        assert not worker.poll_results()

        stats = worker.get_status()["statistics"]

        assert stats["expired_on_poll"] == 1

    finally:

        assert worker.stop()

    print("PASS: Stale buffered result rejection")


def test_frame_ordering():

    worker = AsyncVisionWorker(
        simple_analyser
    )

    worker.start()

    try:

        submit(worker, 5)

        assert worker.wait_until_idle(2.0)

        rejected = False

        try:

            submit(worker, 4)

        except ValueError:

            rejected = True

        assert rejected

    finally:

        assert worker.stop()

    print("PASS: Frame ordering validation")


def test_shutdown_and_restart():

    worker = AsyncVisionWorker(
        simple_analyser
    )

    worker.start()

    submit(worker, 1)

    assert worker.wait_until_idle(2.0)

    assert worker.stop()

    assert not worker.get_status()["running"]

    rejected = False

    try:

        submit(worker, 2)

    except RuntimeError:

        rejected = True

    assert rejected

    worker.start()

    try:

        submit(worker, 3)

        assert worker.wait_until_idle(2.0)

        results = worker.poll_results()

        assert len(results) == 1

        assert results[0].frame_id == 3

    finally:

        assert worker.stop()

    print("PASS: Worker shutdown and restart")


def test_shutdown_during_inference():

    entered = Event()
    release = Event()

    def analyser(request):

        entered.set()

        if not release.wait(3.0):

            raise TimeoutError("Test gate timed out.")

        return simple_analyser(request)

    worker = AsyncVisionWorker(analyser)

    worker.start()

    try:

        submit(worker, 1)

        assert entered.wait(2.0)

        # Running inference cannot be forcibly
        # cancelled by stopping the worker.

        stopped = worker.stop(timeout=0.01)

        assert not stopped

        release.set()

        assert worker.stop(timeout=2.0)

        assert not worker.poll_results()

        assert not worker.get_status()["busy"]

    finally:

        release.set()

        worker.stop()

    print("PASS: Safe shutdown during active inference")


def test_concurrent_perception_simulation():

    def analyser(request):

        time.sleep(0.3)

        return simple_analyser(request)

    worker = AsyncVisionWorker(analyser)

    worker.start()

    perception_iterations = 0

    start = time.monotonic()

    try:

        while time.monotonic() - start < 2.0:

            perception_iterations += 1

            submit(
                worker,
                perception_iterations,
            )

            # Simulate continuous perception
            # operating at approximately 10 Hz.

            time.sleep(0.1)

        elapsed = time.monotonic() - start

        assert worker.wait_until_idle(3.0)

        results = worker.poll_results()

        stats = worker.get_status()["statistics"]

        frequency = (
            perception_iterations / elapsed
        )

        assert perception_iterations >= 15

        assert stats["replaced"] > 0

        assert len(results) > 0

        assert (
            results[-1].frame_id
            == perception_iterations
        )

        summary = {
            "perception_iterations": (
                perception_iterations
            ),
            "perception_frequency_hz": round(
                frequency, 3
            ),
            "simulated_inference_ms": 300,
            "requests_submitted": stats["submitted"],
            "requests_processed": stats["started"],
            "requests_replaced": stats["replaced"],
            "results_received": len(results),
            "mean_inference_ms": round(
                sum(
                    result.inference_ms
                    for result in results
                ) / len(results),
                3,
            ),
            "passed": True,
        }

        print(
            "\n========== CONCURRENCY SUMMARY ==========\n"
        )

        print(json.dumps(summary, indent=4))

    finally:

        assert worker.stop()

    print(
        "\nPASS: Concurrent perception simulation"
    )


def main():

    print(
        "\n========== VISION WORKER TESTS ==========\n"
    )

    tests = [
        test_basic_inference,
        test_non_blocking_submission,
        test_latest_frame_wins,
        test_context_invalidation,
        test_automatic_continuity_change,
        test_expired_request,
        test_expired_result,
        test_error_recovery,
        test_invalid_output,
        test_snapshot_isolation,
        test_bounded_result_buffer,
        test_expired_on_poll,
        test_frame_ordering,
        test_shutdown_and_restart,
        test_shutdown_during_inference,
        test_concurrent_perception_simulation,
    ]

    start = time.perf_counter()

    for test in tests:

        test()

    elapsed = time.perf_counter() - start

    print(
        f"\nAll {len(tests)} vision worker tests passed."
    )

    print(
        f"Total test duration: {elapsed:.3f}s"
    )


if __name__ == "__main__":

    main()
