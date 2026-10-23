"""Stage 6 table: standardised substrates, product and condition key per classified reaction."""

import pyarrow as pa

from platepu.standardise.molecule import StdConfig, condition_key, required, standardize

STANDARDISED_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("licence", pa.string()),
        ("class", pa.string()),
        ("substrate_a", pa.string()),
        ("key_a_stereo", pa.string()),
        ("key_a_flat", pa.string()),
        ("substrate_b", pa.string()),
        ("key_b_stereo", pa.string()),
        ("key_b_flat", pa.string()),
        ("product", pa.string()),
        ("key_product_stereo", pa.string()),
        ("condition_key", pa.list_(pa.string())),
        ("n_reactants", pa.int64()),
        ("n_products", pa.int64()),
        ("n_dynamic_bonds", pa.int64()),
        ("std_version", pa.string()),
    ]
)


def standardise_table(classified: pa.Table, config: StdConfig) -> pa.Table:
    """Standardise each classified reaction.

    Substrates and the product are standardised as substrates; the condition key is built from the
    reagent side of the mapped reaction (ADR 002), so it is the same for every source. A reagent that
    cannot be read leaves ``condition_key`` empty (null): the reaction keeps its substrates. A
    substrate or product that fails raises ValueError (``required``).
    """
    rows = []
    for row in classified.to_pylist():
        _, reagents, products = row["clean_rsmi"].split(">")
        record_id = row["record_id"]
        a = required(standardize(row["substrate_a"], "substrate", config), record_id)
        b = required(standardize(row["substrate_b"], "substrate", config), record_id)
        product = required(standardize(products, "substrate", config), record_id)
        condition = condition_key(reagents.split(".") if reagents else (), (), config)
        rows.append(
            {
                **{key: row[key] for key in ("record_id", "licence", "class")},
                **{
                    key: row[key]
                    for key in ("n_reactants", "n_products", "n_dynamic_bonds")
                },
                "substrate_a": a.smiles,
                "key_a_stereo": a.key_stereo,
                "key_a_flat": a.key_flat,
                "substrate_b": b.smiles,
                "key_b_stereo": b.key_stereo,
                "key_b_flat": b.key_flat,
                "product": product.smiles,
                "key_product_stereo": product.key_stereo,
                "condition_key": list(condition) if condition is not None else None,
                "std_version": config.std_version,
            }
        )
    return pa.Table.from_pylist(rows, schema=STANDARDISED_SCHEMA)
