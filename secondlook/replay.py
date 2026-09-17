"""Read-only access to a recorded LeRobot v3 dataset for mission control REPLAY.

Everything served from here is labelled REPLAY. Nothing here touches the robot.
"""
from __future__ import annotations

import json
from pathlib import Path


class ReplayError(ValueError):
    pass


class ReplayDataset:
    def __init__(self, root: Path):
        import pyarrow.parquet as pq

        self.root = Path(root).resolve()
        info = json.loads((self.root / "meta/info.json").read_text())
        if info.get("codebase_version") != "v3.0":
            raise ReplayError(f"Unsupported LeRobot dataset version: {info.get('codebase_version')}")
        self.fps = info["fps"]
        self.robot_type = info.get("robot_type")
        features = info["features"]
        self.joint_names = [name.removesuffix(".pos") for name in features["observation.state"]["names"]]
        if features["action"]["names"] != features["observation.state"]["names"]:
            raise ReplayError("Action and state joint names differ")
        self.cameras = sorted(key for key, value in features.items() if value.get("dtype") == "video")
        self._video_path = info["video_path"]

        episode_rows = []
        for path in sorted((self.root / "meta/episodes").glob("chunk-*/file-*.parquet")):
            episode_rows += pq.read_table(path, columns=self._episode_columns(path, pq)).to_pylist()
        rows = []
        for path in sorted((self.root / "data").glob("chunk-*/file-*.parquet")):
            rows += pq.read_table(path, columns=["episode_index", "frame_index",
                                                 "observation.state", "action"]).to_pylist()
        if len(rows) != info["total_frames"] or len(episode_rows) != info["total_episodes"]:
            raise ReplayError("Dataset row counts disagree with meta/info.json")

        self._episodes: dict[int, dict] = {}
        for episode in sorted(episode_rows, key=lambda row: row["episode_index"]):
            index = episode["episode_index"]
            frames = sorted((row for row in rows if row["episode_index"] == index),
                            key=lambda row: row["frame_index"])
            if len(frames) != episode["length"]:
                raise ReplayError(f"Episode {index} frame count disagrees with its metadata")
            self._episodes[index] = {
                "index": index,
                "task": (episode["tasks"] or [None])[0],
                "length": episode["length"],
                "duration_s": episode["length"] / self.fps,
                "state": [[round(v, 3) for v in row["observation.state"]] for row in frames],
                "action": [[round(v, 3) for v in row["action"]] for row in frames],
                "videos": [self._video(camera, episode) for camera in self.cameras],
            }

    def _episode_columns(self, path: Path, pq) -> list[str]:
        wanted = ("episode_index", "tasks", "length")
        return [name for name in pq.read_schema(path).names
                if name in wanted or name.startswith("videos/")]

    def _video(self, camera: str, episode: dict) -> dict:
        prefix = f"videos/{camera}/"
        chunk, file = episode[prefix + "chunk_index"], episode[prefix + "file_index"]
        return {"camera": camera, "chunk": chunk, "file": file,
                "url": f"/api/replay/video/{self.cameras.index(camera)}/{chunk}/{file}",
                "from_timestamp": episode[prefix + "from_timestamp"],
                "to_timestamp": episode[prefix + "to_timestamp"]}

    def summary(self) -> dict:
        return {"source": "REPLAY", "dataset": self.root.name if self.root.name != "dataset" else self.root.parent.name,
                "robot_type": self.robot_type, "fps": self.fps, "joint_names": self.joint_names,
                "cameras": self.cameras,
                "episodes": [{key: episode[key] for key in ("index", "task", "length", "duration_s")}
                             for episode in self._episodes.values()]}

    def episode(self, index: int) -> dict:
        if index not in self._episodes:
            raise KeyError(index)
        return {"source": "REPLAY", "fps": self.fps, "joint_names": self.joint_names,
                **self._episodes[index]}

    def video_file(self, camera_index: int, chunk: int, file: int) -> Path:
        """Resolve only files some episode references, so no client path reaches the filesystem."""
        known = {(video["camera"], video["chunk"], video["file"])
                 for episode in self._episodes.values() for video in episode["videos"]}
        camera = self.cameras[camera_index] if 0 <= camera_index < len(self.cameras) else None
        if (camera, chunk, file) not in known:
            raise KeyError((camera_index, chunk, file))
        return self.root / self._video_path.format(video_key=camera, chunk_index=chunk, file_index=file)
