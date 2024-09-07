# from rich.progress import Progress
from src.schedulers.pool_scheduler import PoolProcessingScheduler


# scheduler  = PoolProcessingScheduler(2)

# def work(n: int) -> None:
#     import time 
#     time.sleep(n)


# if __name__ == '__main__':
#     N_population = 110
#     # with Progress() as progress:
#     #     task_id = progress.add_task('fitness', total=N_population)
#     # scheduler.run_with_progress([1]*N_population, work, callback=lambda _: progress.advance(task_id))
        
#     scheduler.run_with_progress([1]*N_population, work, callback=lambda _: print('here'))
        


import time
from rich.progress import Progress


def do_work(n):
    time.sleep(1)
    return n


if __name__ == "__main__":
    scheduler = PoolProcessingScheduler(3)

    with Progress() as progress:
        task_id = progress.add_task("[green]Completed...", total=100)
        scheduler.run_with_progress(range(100), do_work, lambda p: progress.update(task_id, completed=p))    