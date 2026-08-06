import psycopg


class DBLogger:
    """Logs predictions to a database."""
    def __init__(self, db_uri: str):
        self.db_uri = db_uri

    def connect(self):
        self.connection = psycopg.connect(self.db_uri)

    def log_image_metadata(self, image_metadata: list[tuple]):
        """Insert image metadata rows into the database.

        Args:
            image_metadata: list of tuples, each:
                (plate, well, field, channel, root_path, file_name, shape_x, shape_y)
        """

        ## TODO: change to multi-row insert
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
        ## TODO: change to multi-row insert
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
        ## TODO: change to multi-row insert
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.executemany("""
                    INSERT INTO tile_stack_member (tile_stack_id, image_id)
                    VALUES (%s, %s) ON CONFLICT (tile_stack_id, image_id) DO NOTHING
                """, tile_stack_members)
                    self.connection.commit()
                    return None
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None
    
    def log_image_prediction(self, image_prediction: tuple):
        """Insert image prediction row into the database.

        Args:
            image_prediction: tuple, each:
                (plate, well, field, run_id, p_label, t_label, total_tiles, vote_fraction, avg_confidence)
        """
        ## TODO: change to multi-row insert
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute("""
                    INSERT INTO image_prediction (plate, well, field, run_id, p_label, t_label, total_tiles, vote_fraction, avg_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                (image_pred_id, tile_stack_id, run_id, p_label, t_label, confidence)
        """
        ## TODO: change to multi-row insert
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.executemany("""
                    INSERT INTO tile_prediction (image_pred_id, tile_stack_id, run_id, p_label, t_label, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, tile_predictions)
                self.connection.commit()
                return None
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None
    