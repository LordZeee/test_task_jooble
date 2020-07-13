# test_task_jooble


usage:  python main.py [-h] [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--chunk_size CHUNK_SIZE]


Evaluation of mean,standard deviation values is complicated in terms of RAM usage, if file size is huge (~10^6 rows). Single PC can't handle it .
The way to deal with that can be Welford's online algorithm ,values computes iteratively  on  data-batches in one pass. 
Unfortunately the accuracy is corrupted because of  averaging. Another way is to use distributed processing like Apache Spark.
