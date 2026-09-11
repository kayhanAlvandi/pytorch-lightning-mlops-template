SELECT * FROM image_prediction;

SELECT * 
FROM tile_prediction t JOIN image_prediction img ON t.image_pred_id = img.id
WHERE img.id = 1;

SELECT t.p_label, COUNT(*) as count
FROM tile_prediction t JOIN image_prediction img ON t.image_pred_id = img.id
WHERE img.id = 2
GROUP BY t.p_label;

SELECT * FROM image_metadata;

SELECT * FROM tile_stack LIMIT 10;

SELECT * FROM tile_stack_member t WHERE t.tile_stack_id = 2 LIMIT 10;

-- list all the tables
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY table_name;

Select * FROM tile_channel_stats;

Select * FROM reference_image_prediction;
