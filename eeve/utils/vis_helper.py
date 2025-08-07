from rich import box
from rich.console import Console
from rich.table import Table


class VisHelper:
    def __init__(self, stats):
        self.stats = stats
    
    def print_rich_table(self):
        console = Console()
        table = Table(title="Statistics", box=box.ROUNDED)
        
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        stats_dict = self.stats.to_dict()
        
        for key, value in stats_dict.items():
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, float):
                formatted_value = f"{value:,.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            elif isinstance(value, list):
                formatted_value = f"[{', '.join(str(v) for v in value)}]"
            elif isinstance(value, dict):
                formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            table.add_row(formatted_key, formatted_value)
        
        console.print(table)