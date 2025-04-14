from e3nn import o3
from mace.modules.blocks import tp_out_irreps_with_instructions

from threedscriptors.utils.model_utils import get_invariant_indices

input_irreps = "128x0e+128x1o+128x0e"
embedding_dim = 10

_, invariant_irreps = get_invariant_indices(o3.Irreps(input_irreps))
i_in1, permut, _ = o3.Irreps(input_irreps).sort()
i_in1 = i_in1.simplify()


embedd_irrep1 = o3.Irreps(f"{embedding_dim}x1o")
embedd_irrep2 = o3.Irreps(f"{embedding_dim}x1e")


lin0 = o3.Linear(i_in1, embedd_irrep1)


# This creates a even vector valued irrep that can be combined with
tp_1 = o3.TensorProduct(
    i_in1,
    embedd_irrep1,
    o3.Irreps("128x1e"),
    instructions=[(1, 0, 0, "uvu", True)],
    shared_weights=True,
    internal_weights=True,
)

lin = o3.Linear(tp_1.irreps_out, embedd_irrep2)

print(lin.irreps_out)

irreps_mid, instructions = tp_out_irreps_with_instructions(
    i_in1,
    embedd_irrep2,
    o3.Irreps("1x0o"),
)


tp_2 = o3.TensorProduct(
    i_in1,
    lin.irreps_out,
    irreps_mid,
    instructions=instructions,
    shared_weights=True,
    internal_weights=True,
)
