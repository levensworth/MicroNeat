from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from rich.live import Live
from rich.table import Table
from rich.console import Console


from src.genome import Genome
from src.population import Population

import typing

if typing.TYPE_CHECKING:
    from src.misc.callbacks.history_callback import History


_HISTORY_TABLE_KEYS = [
    "best_fitness",
    "avg_fitness",
    "weight_mutation_change",
    "weight_perturbation_pc",
    "max_hidden_nodes",
    "max_hidden_connections",
]


class EvolutionVisualizer:
    """Helper class which handles the logic to create a terminal UI dashboard
    which displays some key information of the evolution process.
    """

    def __init__(self, population: Population) -> None:
        self.population = population

    def run(
        self, generations: int, fitness_function: typing.Callable[[Genome], float]
    ) -> Population:
        self.console = Console()

        with Live(
            self.render_dashboard(), console=self.console, refresh_per_second=4
        ) as live:
            self.population.evolve(
                generations=generations,
                fitness_function=fitness_function,
                func=lambda hist: live.update(self.render_dashboard(hist)),
            )

        return self.population

    def render_dashboard(self, history: typing.Optional["History"] = None) -> Layout:
        """Renders a new Dashboard layout based on rich terminal ui.

        Args:
            history (typing.Optional[&#39;History&#39;], optional): The History Callback
                instance of the population we want to print. Defaults to None.

        Returns:
            Layout: Layout instance ready for preview.
        """
        layout = Layout()

        # Divide the "screen" in to three parts
        layout.split(
            Layout(
                Panel(
                    Text("Welcome to MicroNeat! Hope you have fun 😎"),
                    title="MicroNeat",
                    title_align="center",
                ),
                name="header",
                size=3,
            ),
            Layout(ratio=1, name="main"),
            Layout(
                Panel(Text("Made with ❤️ by Levensworth", justify="center")),
                size=3,
                name="footer",
            ),
        )
        # Divide the "main" layout in to "side" and "body"
        layout["main"].split(
            Layout(name="side"), Layout(name="body", ratio=2), splitter="row"
        )

        # set tabular data on for the main body
        if history:
            layout["body"].update(
                Panel(self.create_history_table(history=history), title="stats")
            )

        # Divide the "side" layout in to two
        layout["side"].split(
            Layout(name="current generation"), Layout(name="best generation")
        )
        if history:
            layout["current generation"].update(
                Panel(
                    history._current_generation_processing, title="current generation"
                )
            )

            layout["best generation"].update(
                Panel(
                    Text(
                        str(history.history.get("best_genome", ["not started"])[-1]),
                        justify="center",
                    ),
                    title="best genome",
                    title_align="center",
                )
            )
        return layout

    def create_history_table(self, history: "History") -> Table:
        table = Table(expand=True)
        for col_name in ["NAME", "CURRENT", "PAST", "INCREASE", "INCREASE (%)"]:
            table.add_column(col_name)
        if history._current_generation < 2:
            return table

        for key, list_val in history.history.items():
            if key not in _HISTORY_TABLE_KEYS:
                continue
            change = float(list_val[-1]) - float(list_val[-2]) if float(list_val[-2]) else float(list_val[-1])
            change_percentage = (change / float(list_val[-2])) * 100 if float(list_val[-2]) != 0 else 100

            style = "bold green4" if change > 0 else "bold magenta"

            table.add_row(
                key,
                str(round(float(list_val[-1]), 2)),
                str(round(float(list_val[-2]), 2)),
                Text.styled(str(round(change)) + "%", style=style),
                Text.styled(str(round(change_percentage)) + "%", style=style),
            )

        return table
