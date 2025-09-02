Hi! This is Ose!


# Python
__pycache__/
*.py[cod]
*.egg-info/
venv/
env/
.Python

# data
*.xlsx
data_inputs/
outputs/

# OS
.DS_Store
Thumbs.db



import click
from .pipeline import run_pipeline

@click.command()
@click.option("--input", "input_csv", required=True, help="Path to input CSV from enumerators")
@click.option("--output", "output_csv", required=True, help="Path to output cleaned CSV")
@click.option("--isco", "isco_xlsx", required=True, help="Path to ISCO Excel file")
@click.option("--isic", "isic_xlsx", required=True, help="Path to ISIC Excel file")
@click.option("--config", "config_path", default=None, help="Optional YAML config path")
def main(input_csv, output_csv, isco_xlsx, isic_xlsx, config_path):
    """
    NBS ISCO & ISIC Cleaning Tool (CLI)
    """
    df, review_path = run_pipeline(
        input_csv=input_csv,
        output_csv=output_csv,
        isco_xlsx=isco_xlsx,
        isic_xlsx=isic_xlsx,
        config_path=config_path
    )
    click.echo(f"Saved cleaned file to: {output_csv}")
    if review_path:
        click.echo(f"Saved review queue to: {review_path}")


if __name__ == "__main__":
    main()




python -m nbs_code_cleaner.cli \
  --input data_inputs/sample_input.csv \
  --output outputs/cleaned_sample.csv \
  --isco "/path/to/ISCO-08 EN Structure and definitions.xlsx" \
  --isic "/path/to/ISIC Rev. 5 Explanatory Notes.xlsx" \
  --config nbs_code_cleaner/config_default.yaml   # optional; omit to use packaged default



Job_Title,Job_Description,Job_Sector,Enumerator_ISCO,Enumerator_ISIC
Secondary school teacher,Teaches mathematics to teenagers,Education,2330,8520
Taxi driver,Drives taxi in Lagos,Transport,8322,4922
Cassava farmer,Works on cassava farm,Agriculture,,0111
Okada rider,Transports passengers on motorcycle,Transport,8322,4922

