// Written by Yi He
// University of Chicago
// This is the injection testbench for NVDLA

// Define Input Vector

module tb_inject();

//initial begin
    // File Path
    //$vcdpluson(0, top.nvdla_top);
//end

// Give the clock signal to NVDLA's core clock
wire clock_vector = top.nvdla_top.dla_core_clk;

// Define Mask


always @ (posedge clock_vector) begin
    // Insert Force

    $display("CNT %d\n", top.nvdla_top.counter);
end

endmodule
