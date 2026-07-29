"""
DBSCAN clustering nepoznatih lica.
Automatski grupira nepoznate osobe bez poznavanja
broja klastera unaprijed.

Filtriranje na ulazu:
- best_known_score < 0.3  → nije poznata osoba u lošem kutu
- face_score >= 0.0       → lice postoji
- det_score > 0.65        → dobra kvaliteta detekcije

DBSCAN parametri:
- eps = 0.4        → udaljenost između točaka
- min_samples = 2  → minimalno 2 slične pojave = klaster
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.cluster import DBSCAN

CLUSTERS_PATH = Path(__file__).parent / "unknown_clusters.json"
KNOWN_EXCLUSION_THRESHOLD = 0.3


class UnknownPersonClustering:

    def __init__(self, eps: float = 0.4, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples

        self._embeddings = []
        self._timestamps = []
        self._det_scores = []

        self._clusters = {}
        self._labels = []

        self._load_clusters()

    def should_add(
        self,
        embedding: np.ndarray,
        face_score: float,
        det_score: float,
        known_persons: dict,
    ) -> bool:
        """
        Provjeri treba li dodati ovaj embedding u clustering.

        Uvjeti:
        1. Lice postoji (face_score >= 0.0)
        2. InsightFace je siguran (det_score > 0.65)
        3. Nije poznata osoba u lošem kutu
           (best_known_score < KNOWN_EXCLUSION_THRESHOLD)
        """
        if face_score < 0.0:
            return False

        if det_score < 0.65:
            return False

        if known_persons:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                emb_norm = embedding / norm
            else:
                emb_norm = embedding

            best_known = max(
                float(
                    np.dot(emb_norm, db_emb)
                    / (np.linalg.norm(emb_norm) * np.linalg.norm(db_emb))
                )
                for db_emb in known_persons.values()
            )

            if best_known >= KNOWN_EXCLUSION_THRESHOLD:
                return False

        return True

    def add_unknown(self, embedding: np.ndarray, det_score: float = 1.0):
        """Dodaj embedding nepoznate osobe u buffer i pokreni clustering."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self._embeddings.append(embedding)
        self._timestamps.append(datetime.now().isoformat())
        self._det_scores.append(float(det_score))

        if len(self._embeddings) >= self.min_samples:
            self._run_clustering()

    def _run_clustering(self):
        """Pokreni DBSCAN na svim prikupljenim embeddingsima."""
        if len(self._embeddings) < 2:
            return

        X = np.array(self._embeddings)

        db = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric="cosine", n_jobs=-1
        )
        self._labels = db.fit_predict(X)

        unique_labels = set(self._labels)
        new_clusters = {}

        for label in unique_labels:
            if label == -1:
                continue

            indices = [i for i, l in enumerate(self._labels) if l == label]

            cluster_embeddings = X[indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            existing = self._clusters.get(int(label), {})

            new_clusters[int(label)] = {
                "cluster_id": int(label),
                "count": int(len(indices)),
                "centroid": centroid.tolist(),
                "first_seen": existing.get("first_seen", self._timestamps[indices[0]]),
                "last_seen": self._timestamps[indices[-1]],
            }

        self._clusters = new_clusters
        self._save_clusters()

    def identify_unknown(self, embedding: np.ndarray) -> tuple:
        """
        Provjeri pripada li novi embedding postojećem klasteru.

        Vraća:
          (cluster_id, similarity) → poznat klaster
          (None, score)            → novi/outlier
        """
        if not self._clusters:
            return None, -1.0

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        best_id = None
        best_score = -1.0

        for cluster_id, info in self._clusters.items():
            centroid = np.array(info["centroid"])
            score = float(np.dot(embedding, centroid))
            if score > best_score:
                best_score = score
                best_id = int(cluster_id)

        if best_score > (1 - self.eps):
            return best_id, float(best_score)
        return None, float(best_score)

    def get_clusters(self) -> list:
        """Vrati listu svih klastera sortiranu po broju pojavljivanja."""
        return sorted(self._clusters.values(), key=lambda x: x["count"], reverse=True)

    def get_stats(self) -> dict:
        """Statistike clusteringa."""
        n_outliers = int(sum(1 for l in self._labels if l == -1))
        return {
            "total_embeddings": int(len(self._embeddings)),
            "num_clusters": int(len(self._clusters)),
            "num_outliers": n_outliers,
            "eps": float(self.eps),
            "min_samples": int(self.min_samples),
            "known_exclusion_threshold": float(KNOWN_EXCLUSION_THRESHOLD),
        }

    def reset(self):
        """Resetiraj sve podatke."""
        self._embeddings = []
        self._timestamps = []
        self._det_scores = []
        self._clusters = {}
        self._labels = []
        self._save_clusters()
        print("✅ Clustering resetiran")

    def _save_clusters(self):
        data = {
            "clusters": {str(k): v for k, v in self._clusters.items()},
            "stats": self.get_stats(),
            "updated": datetime.now().isoformat(),
        }
        with open(CLUSTERS_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def _load_clusters(self):
        if CLUSTERS_PATH.exists():
            with open(CLUSTERS_PATH) as f:
                data = json.load(f)
            # Konvertiraj ključeve natrag u int
            self._clusters = {int(k): v for k, v in data.get("clusters", {}).items()}
            print(f"✅ Clustering: učitano {len(self._clusters)} klastera")
