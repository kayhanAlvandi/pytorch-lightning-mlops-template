import psycopg


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
                (tile_stack_id, img_id)
        """
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.executemany("""
                    INSERT INTO tile_stack_member (tile_stack_id, image_id)
                    VALUES (%s, %s) ON CONFLICT (tile_stack_id, image_id) DO UPDATE SET tile_stack_id = EXCLUDED.tile_stack_id, image_id = EXCLUDED.image_id
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