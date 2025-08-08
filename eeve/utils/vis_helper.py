from rich import box
from rich.console import Console
from rich.table import Table


class VisHelper:
    def __init__(self, title: str = "Statistics", col1_color: str = "cyan", col2_color: str = "magenta"):
        self.title = title
        self.col1_color = col1_color
        self.col2_color = col2_color

    @staticmethod
    def _format_key(key) -> str:
        key_str = str(key)
        return key_str.replace('_', ' ').title()

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, float):
            return f"{value:,.2f}"
        elif isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, list):
            return f"[{', '.join(str(v) for v in value)}]"
        else:
            return str(value)

    def print_rich_table(self, stats) -> None:
        console = Console()
        
        for tokenizer_name, stat_storage in stats.items():
            table = Table(title=f"{self.title} - {tokenizer_name}", box=box.ROUNDED)
            table.add_column("Parameter", style=self.col1_color, no_wrap=True)
            table.add_column("Value", style=self.col2_color)
            
            stats_dict = stat_storage.to_dict()
            for key, value in stats_dict.items():
                table.add_row(self._format_key(key), self._format_value(value))
            
            console.print(table)

    def print_comparisons(self, stats_dict) -> None:
        console = Console()

        if not stats_dict:
            console.print("[bold red]No statistics provided for comparison.[/]")
            return
        
        tokenizer_names = list(stats_dict.keys())
        stats_list_values = list(stats_dict.values())
        
        ordered_keys = []
        seen = set()
        for s in stats_list_values:
            for k in s.to_dict().keys():
                if k not in seen:
                    seen.add(k)
                    ordered_keys.append(k)
        
        table = Table(title=self.title, box=box.ROUNDED)
        table.add_column("Parameter", style=self.col1_color, no_wrap=True)
        
        for name in tokenizer_names:
            table.add_column(name, style=self.col2_color)
        
        stats_dicts = [s.to_dict() for s in stats_list_values]
        
        for key in ordered_keys:
            row = [self._format_key(key)]
            for d in stats_dicts:
                val = d.get(key, "—")
                row.append(self._format_value(val) if val != "—" else "—")
            table.add_row(*row)
        
        console.print(table)