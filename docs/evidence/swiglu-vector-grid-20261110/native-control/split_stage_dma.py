"""Native two-stage traffic controls for a four-row FP32 tile with a three-row tail."""

SCHEDULES = ("single", "serial", "early")
MODES = tuple(name + suffix for name in SCHEDULES for suffix in ("", "_upper"))


def render(prologue, mode):
    """Use the qualified grid, input policy and slab from the full DMA control."""
    schedule = mode.removesuffix("_upper")
    if schedule not in SCHEDULES:
        raise ValueError(mode)
    # First two rows always exist; the second stage has either one or two.
    # Both slot origins, input lengths, output lengths and UB offsets are
    # multiples of 32 bytes even for the last block.
    upper = mode.endswith("_upper")
    offset_a = "2 * H" if upper else "0"
    offset_b = "4 * H + second * H" if upper else "4 * H"
    setup = prologue + """
  const int32_t second = valid - 2;
  DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};
"""
    def read(local, gm, words):
        return f"""  DataCopyPad(slab[{local}], src[{gm}],
      DataCopyExtParams{{1, static_cast<uint32_t>(({words}) * sizeof(float)), 0, 0, 0}}, pad);
"""
    def write(local, gm, words):
        return f"""  DataCopyPad(dst[{gm}], slab[{local}],
      DataCopyExtParams{{1, static_cast<uint32_t>(({words}) * sizeof(float)), 0, 0, 0}});
"""
    def event(action, path, event_id):
        return f"  {action}Flag<HardEvent::{path}>(EVENT_ID{event_id});\n"

    read_a = read("0", "row * 2 * H", "4 * H")
    read_b = read("4 * H", "(row + 2) * 2 * H", "2 * second * H")
    write_a = write(offset_a, "row * H", "2 * H")
    write_b = write(offset_b, "(row + 2) * H", "second * H")
    ready_a = event("Set", "MTE2_MTE3", 0)
    ready_b = event("Set", "MTE2_MTE3", 1)
    wait_a = event("Wait", "MTE2_MTE3", 0)
    wait_b = event("Wait", "MTE2_MTE3", 1)
    if schedule == "single":
        instructions = read("0", "row * 2 * H", "valid * 2 * H") + ready_a + wait_a + write_a + write_b
    elif schedule == "early":
        instructions = read_a + ready_a + read_b + ready_b + wait_a + write_a + wait_b + write_b
    else:
        drain_a = event("Set", "MTE3_MTE2", 0) + event("Wait", "MTE3_MTE2", 0)
        instructions = read_a + ready_a + wait_a + write_a + drain_a + read_b + ready_b + wait_b + write_b
    return setup + instructions + "}\n"
