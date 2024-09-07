import time
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from rich.live import Live
from rich.table import Table
from rich.console import Console

import typing
if typing.TYPE_CHECKING:
    from src.misc.callbacks.history_callback import History



_HISTORY_TABLE_KEYS = ['best_fitness', 'avg_fitness', 'weight_mutation_change', 'weight_perturbation_pc', 'max_hidden_nodes', 'max_hidden_connections']

def create_history_table(history: 'History') -> Table:
    table = Table(expand=True)
    for col_name in ["NAME", "CURRENT", "PAST", "INCREASE", "INCREASE (%)"]:
        table.add_column(col_name)
    if history._current_generation < 2:
        return table
    
    for key, list_val in history.history.items():
        if key not in  _HISTORY_TABLE_KEYS:
            continue
        change = list_val[-1] - list_val[-2]
        change_percentage = (change / float(list_val[-2])) * 100
        
        style = 'bold green4' if change > 0 else 'bold magenta'

        table.add_row(
            key, 
            str(round(list_val[-1])), 
            str(round(list_val[-2])), 
            Text.styled(str(round(change)) + '%', style=style), 
            Text.styled(str(round(change_percentage)) + '%', style=style)
        )
    
    return table






def render_dashboard(history: typing.Optional['History'] = None):
    layout = Layout()

    # Divide the "screen" in to three parts
    layout.split(
        Layout(Panel(Text('Welcome to MicroNeat! Hope you have fun 😎'), title='MicroNeat', title_align='center'), name="header", size=3),
        Layout(ratio=1, name="main"),
        Layout(Panel(Text('Made with ❤️ by Levensworth', justify='center')), size=3, name="footer"),
    )
    # Divide the "main" layout in to "side" and "body"
    layout["main"].split(
        Layout(name="side"),
        Layout(name="body", ratio=2),
        splitter="row"
    )


    # set tabular data on for the main body
    if history:
        layout['body'].update(
            Panel(
                create_history_table(history=history),
                title='stats'
            )
            
        )

    # Divide the "side" layout in to two
    layout["side"].split(Layout(name='current generation'), Layout(name='best generation'))
    if history:
        layout['current generation'].update(
            Panel(
                history._current_generation_processing,
                title='current generation'
            )
        )

        layout['best generation'].update(
            Panel(
                Text(
                    str(history.history.get('best_genome', ['not started'])[-1]),
                    justify='center'
                ),
                title='best genome',
                title_align='center'
            )
        )
    return layout

import gymnasium as gym

from gyn_fitness import GymFitnessFunction
from src.population import Population


def make_env():
    """ Makes a new gym 'CartPole-v1' environment. """
    return gym.make("CartPole-v1")

if __name__ == '__main__':
    fitness_function = GymFitnessFunction(make_env=make_env, default_num_episodes=5, default_max_steps=1000)
    population = Population(size=100, n_inputs=4, n_outputs=2,with_bias=True)
    
    console = Console()

    with Live(render_dashboard(), console=console,  refresh_per_second=4) as live:
        population.evolve(
        generations=20,
        fitness_function=fitness_function,
        func=lambda hist: live.update(render_dashboard(hist))
        )
    
        