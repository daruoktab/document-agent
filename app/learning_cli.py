"""Admin-only local commands: python -m app.learning_cli --help."""

from __future__ import annotations

import argparse
import json

from .learning_store import LearningStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kelola pembelajaran dari koreksi dokumen."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "status", help="Lihat jumlah koreksi, versi aktif, dan percobaan"
    )
    commands.add_parser(
        "probe", help="Uji satu gambar buatan melalui model/API yang dikonfigurasi"
    )
    optimize_parser = commands.add_parser(
        "optimize", help="Jalankan GEPA; memakai model/API yang dikonfigurasi"
    )
    optimize_parser.add_argument("--max-metric-calls", type=int, default=50)
    activate_parser = commands.add_parser(
        "activate", help="Terapkan kandidat yang lulus evaluasi"
    )
    activate_parser.add_argument("run_id")
    commands.add_parser("rollback", help="Pulihkan versi sebelumnya")
    args = parser.parse_args()
    store = LearningStore()
    try:
        if args.command == "status":
            print(
                json.dumps(
                    {
                        "ready_examples": len(store.examples()),
                        "active": store.active(),
                        "runs": store.runs(),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif args.command == "rollback":
            print(f"Versi dipulihkan: {store.rollback()}")
        else:
            from .dspy_learning import model_fingerprint, optimize, probe_model

            if args.command == "probe":
                print(probe_model())
            elif args.command == "activate":
                store.activate(args.run_id, model_fingerprint())
                print(f"Versi diterapkan untuk ekstraksi baru: {args.run_id}")
            else:
                result = optimize(store, args.max_metric_calls)
                print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 -- CLI boundary; failed runs never publish.
        print(f"Proses gagal ({type(exc).__name__}). Prompt aktif tidak diubah.")
        if isinstance(exc, ValueError):
            print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
