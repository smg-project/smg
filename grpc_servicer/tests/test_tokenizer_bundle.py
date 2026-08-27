import zipfile

from smg_grpc_servicer.tokenizer_bundle import build_tokenizer_zip


def test_build_tokenizer_zip_includes_processor_config(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "processor_config.json").write_text(
        '{"image_processor": {}, "video_processor": {}}', encoding="utf-8"
    )

    bundle = build_tokenizer_zip(tmp_path)

    with zipfile.ZipFile(bundle) as archive:
        assert "processor_config.json" in archive.namelist()
