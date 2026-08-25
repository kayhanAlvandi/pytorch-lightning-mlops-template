import psycopg
from psycopg.rows import dict_row


class DBLogger:
    """Logs predictions to a database."""
    def __init__(self, db_uri: str):
        self.db_uri = db_uri

    def connect(self):
        self.connection = psycopg.connect(self.db_uri)
    
    def close(self):
        self.connection.close()

    def log_image_metadata(self, image_metadata: list[tuple]):
        """Insert image metadata rows into the database.

        Args:
            image_metadata: list of tuples, each:
                (plate, well, field, channel, root_path, file_name, shape_x, shape_y)
        """

        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.executemany("""
                    INSERT INTO image_metadata (plate, well , field, channel , root_path
                    ,file_name, shape_x, shape_y)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (file_name) DO UPDATE SET root_path = EXCLUDED.root_path
                    RETURNING id
                """, image_metadata, returning=True)
                    img_ids = []
                    for _ in cursor.results():
                        img_ids.append(cursor.fetchone()[0])
                self.connection.commit()
                return img_ids
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None

    def log_tile_stack(self, tile_stack_metadata: list[tuple]):
        """Insert tile stack metadata rows into the database.

        Args:
            tile_stack_metadata: list of tuples, each:
                (stack_hash, row_ind, col_ind, x_left, y_top, crop_size)
        """
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.executemany("""
                    INSERT INTO tile_stack (stack_hash, row_ind, col_ind, x_left, y_top, crop_size)
                    VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (stack_hash) DO UPDATE SET row_ind = EXCLUDED.row_ind, col_ind = EXCLUDED.col_ind, x_left = EXCLUDED.x_left, y_top = EXCLUDED.y_top, crop_size = EXCLUDED.crop_size
                    RETURNING id
                """, tile_stack_metadata, returning=True)
                    tile_stack_ids = []
                    for _ in cursor.results():
                        tile_stack_ids.append(cursor.fetchone()[0])
                self.connection.commit()
                return tile_stack_ids
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None
    
    def log_tile_stack_member(self, tile_stack_members: list[tuple]):
        """Insert tile stack member rows into the database.

        Args:
            tile_stack_members: list of tuples, each:
                (tile_stack_id, img_id, channel_index)
        """
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.executemany("""
                    INSERT INTO tile_stack_member (tile_stack_id, image_id, channel_index)
                    VALUES (%s, %s, %s) ON CONFLICT (tile_stack_id, image_id)
                        DO UPDATE SET channel_index = EXCLUDED.channel_index
                    RETURNING id
                """, tile_stack_members, returning=True)
                    tile_stack_member_ids = []
                    for _ in cursor.results():
                        tile_stack_member_ids.append(cursor.fetchone()[0])
                self.connection.commit()
                return tile_stack_member_ids
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None
        return None
    
    def log_image_prediction(self, image_prediction: tuple):
        """Insert image prediction row into the database.

        Args:
            image_prediction: tuple, each:
                (plate, well, field, run_id, p_label, t_label, total_tiles, vote_fraction, avg_confidence, is_reference)
        """
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute("""
                    INSERT INTO image_prediction (plate, well, field, run_id, p_label, t_label, total_tiles, vote_fraction, avg_confidence, is_reference)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, image_prediction)
                    image_pred_id = cursor.fetchone()[0]
                self.connection.commit()
                return image_pred_id
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None
    
    def log_tile_prediction(self, tile_predictions: list[tuple]):
        """Insert tile prediction rows into the database.

        Args:
            tile_predictions: list of tuples, each:
                (image_pred_id, tile_stack_id, run_id, p_label, t_label, confidence, is_reference)
        """
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.executemany("""
                    INSERT INTO tile_prediction (image_pred_id, tile_stack_id, run_id, p_label, t_label, confidence, is_reference)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, tile_predictions)
                self.connection.commit()
                return len(tile_predictions) # return number of rows inserted
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None
    
    def log_tile_channel_stats(self, tile_channel_stats: list[tuple]):


        """Insert tile channel stats rows into the database.

        Args:
            tile_channel_stats: list of tuples, each:
                (tile_stack_member_id, mean, std, p1, p5, p95, p99)
        """
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.executemany("""
                    INSERT INTO tile_channel_stats (tile_stack_member_id, mean, std, p1, p5, p95, p99)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (tile_stack_member_id) DO UPDATE SET
                        mean = EXCLUDED.mean,
                        std = EXCLUDED.std,
                        p1 = EXCLUDED.p1,
                        p5 = EXCLUDED.p5,
                        p95 = EXCLUDED.p95,
                        p99 = EXCLUDED.p99
                        RETURNING id
                    """, tile_channel_stats, returning=True)
                    tile_channel_stats_ids = []
                    for _ in cursor.results():
                        tile_channel_stats_ids.append(cursor.fetchone()[0])
                self.connection.commit()
                return tile_channel_stats_ids
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None
    
    def get_reference_samples(self, run_id: str) -> list[tuple]:
        """Return the (plate, well, field) of every validation sample already logged as
        reference for this run_id, so reference computation can resume and skip only the
        samples it already has instead of redoing (or fully skipping) the whole run.

        Args:
            run_id: MLflow run id to check

        Returns:
            List of (plate, well, field) tuples for reference rows that already exist for
            this run
        """
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(""" 
                    SELECT plate, well, field
                    FROM image_prediction
                    WHERE run_id = %s AND is_reference = TRUE
                    """, (run_id,))
                    result = cursor.fetchall()
                    return result
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None

    # ── Drift reporting: reads for reference vs. current comparison ─────────

    def _fetch_dicts(self, query: str, params: tuple) -> list[dict]:
        """Run a read query and return rows as a list of dicts (column -> value)."""
        for attempt in range(2):  # try once, reconnect and retry once
            try:
                with self.connection.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(query, params)
                    return cursor.fetchall()
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None

    def fetch_reference_image_level(self, run_id: str) -> list[dict]:
        """Image-level reference rows for a run: (p_label, vote_fraction, avg_confidence)."""
        return self._fetch_dicts("""
            SELECT p_label, vote_fraction, avg_confidence
            FROM reference_image_prediction
            WHERE run_id = %s
        """, (run_id,))

    def fetch_current_image_level(self, run_id: str, window_start, window_end) -> list[dict]:
        """Image-level production rows in [window_start, window_end) for a run.

        Excludes wells that belong to this run's reference set, so the drift
        window is never compared partly against the reference itself (a
        validation well can legitimately be re-imaged and predicted in prod).
        """
        return self._fetch_dicts("""
            SELECT l.p_label, l.vote_fraction, l.avg_confidence
            FROM live_image_prediction l
            WHERE l.run_id = %s
              AND l.created_at >= %s
              AND l.created_at <  %s
              AND NOT EXISTS (
                  SELECT 1 FROM reference_image_prediction r
                  WHERE r.run_id = l.run_id
                    AND r.plate  = l.plate
                    AND r.well   = l.well
                    AND r.field  = l.field
              )
        """, (run_id, window_start, window_end))

    def fetch_reference_tile_level(self, run_id: str) -> list[dict]:
        """Tile-level reference rows + per-channel input stats for a run.

        One row per (tile_prediction, channel): tile_pred_id and confidence
        repeat across the tile's channels; caller dedups for confidence and
        pivots the channel stats long -> wide.
        """
        return self._fetch_dicts("""
            SELECT t.id AS tile_pred_id,
                   t.p_label,
                   t.confidence,
                   im.channel AS channel,
                   s.mean, s.std, s.p1, s.p5, s.p95, s.p99
            FROM reference_tile_prediction t
            JOIN tile_stack_member tsm     ON tsm.tile_stack_id = t.tile_stack_id
            JOIN image_metadata im         ON im.id = tsm.image_id
            LEFT JOIN tile_channel_stats s ON s.tile_stack_member_id = tsm.id
            WHERE t.run_id = %s
        """, (run_id,))

    def fetch_current_tile_level(self, run_id: str, window_start, window_end) -> list[dict]:
        """Tile-level production rows + per-channel input stats in the window.

        Same reference-well exclusion as fetch_current_image_level, applied via
        the parent image_prediction row.
        """
        return self._fetch_dicts("""
            SELECT t.id AS tile_pred_id,
                   t.p_label,
                   t.confidence,
                   im.channel AS channel,
                   s.mean, s.std, s.p1, s.p5, s.p95, s.p99
            FROM live_tile_prediction t
            JOIN live_image_prediction l   ON l.id = t.image_pred_id
            JOIN tile_stack_member tsm     ON tsm.tile_stack_id = t.tile_stack_id
            JOIN image_metadata im         ON im.id = tsm.image_id
            LEFT JOIN tile_channel_stats s ON s.tile_stack_member_id = tsm.id
            WHERE t.run_id = %s
              AND t.created_at >= %s
              AND t.created_at <  %s
              AND NOT EXISTS (
                  SELECT 1 FROM reference_image_prediction r
                  WHERE r.run_id = l.run_id
                    AND r.plate  = l.plate
                    AND r.well   = l.well
                    AND r.field  = l.field
              )
        """, (run_id, window_start, window_end))

    # ── Drift reporting: writes ──────────────────────────────────────────────

    def log_drift_report(self, drift_report: tuple):
        """Insert one drift-report row and return its id.

        Args:
            drift_report: tuple
                (run_id, window_start, window_end, dataset_drift,
                 n_columns_drifted, n_columns_total, report_path)
        """
        for attempt in range(2):  # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute("""
                    INSERT INTO drift_report
                        (run_id, window_start, window_end, dataset_drift,
                         n_columns_drifted, n_columns_total, report_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, drift_report)
                    drift_report_id = cursor.fetchone()[0]
                self.connection.commit()
                return drift_report_id
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None

    def log_drift_report_column(self, drift_report_columns: list[tuple]):
        """Insert per-column drift results for a drift report.

        Args:
            drift_report_columns: list of tuples, each:
                (drift_report_id, column_name, column_group, drift_score,
                 drifted, stat_test)
        """
        for attempt in range(2):  # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.executemany("""
                    INSERT INTO drift_report_column
                        (drift_report_id, column_name, column_group,
                         drift_score, drifted, stat_test)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, drift_report_columns)
                self.connection.commit()
                return len(drift_report_columns)
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None