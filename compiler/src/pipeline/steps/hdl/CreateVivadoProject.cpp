#include "CreateVivadoProject.h"

#include "EmbedController.hpp"

#include "option/GlobalOptions.h"
#include "circt/Conversion/ExportVerilog.h"

#include "spdlog/fmt/fmt.h"
#include "util/Command.h"
#include "toolchain/MLIRToolchain.h"
#include <fstream>
#include <regex>


namespace spnc {

// TODO: Vivado does not seem to accept underscores in AXI signal names!

static const char TCL_PREAMBLE[] = R"(
set project_name {projectName}
set files [list {fileList}]
set vendor de.tu-darmstadt.esa
set library chisel-if
set litedata {liteDataWidth}
set liteaddr {liteAddrWidth}
set fulladdr {memAddrWidth}
set fulldata {memDataWidth}
set controllerInWidth {controllerInWidth}
set controllerOutWidth {controllerOutWidth}
set version {version}
)";

static const char TCL_PACKAGE[] = R"(

create_project -force $project_name ./$project_name
set project_dir [get_property directory [current_project]]
if {[string equal [get_filesets -quiet sources_1] ""]} {
  create_fileset -srcset sources_1
}

foreach {file} $files {
    add_files -fileset sources_1 $file
}
update_compile_order -fileset sources_1

create_bd_design $project_name
set M_AXI [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI]
set_property CONFIG.DATA_WIDTH $fulldata [get_bd_intf_ports $M_AXI]
set_property CONFIG.ADDR_WIDTH $fulladdr [get_bd_intf_ports $M_AXI]
set S_AXI_LITE [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI_LITE]
set_property CONFIG.DATA_WIDTH $litedata [get_bd_intf_ports $S_AXI_LITE]
set_property CONFIG.ADDR_WIDTH $liteaddr [get_bd_intf_ports $S_AXI_LITE]
set_property CONFIG.PROTOCOL AXI4LITE [get_bd_intf_ports $S_AXI_LITE]
#set AXI4_STREAM_MASTER [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 AXI4_STREAM_MASTER]
#set AXI4_STREAM_SLAVE  [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 AXI4_STREAM_SLAVE]
#set_property CONFIG.TDATA_NUM_BYTES $streamin_bytes [get_<bd_intf_ports $AXI4_STREAM_SLAVE]
#set_property CONFIG.TDATA_NUM_BYTES $streamout_bytes [get_bd_intf_ports $AXI4_STREAM_MASTER]

proc create_inverter {name} {
      variable ret [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 $name]
      set_property -dict [list CONFIG.C_SIZE {1} CONFIG.C_OPERATION {not} CONFIG.LOGO_FILE {data/sym_notgate.png}] [get_bd_cells $name]
      return $ret
    }

set ap_clk   [ create_bd_port -dir I -type clk ap_clk ]
set ap_rst_n [ create_bd_port -dir I -type rst ap_rst_n ]
set out_inv  [ create_inverter reset_inverter]
set interrupt [ create_bd_port -dir O -type intr interrupt ]

set_property CONFIG.POLARITY ACTIVE_LOW $ap_rst_n

set TAP [create_bd_cell -type module -reference AXI4StreamMapper TAP]
set_property CONFIG.POLARITY ACTIVE_HIGH [get_bd_pins $TAP/reset]

set DWCS [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:1.1 DWCS]
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER] [get_bd_cells $DWCS]
set_property CONFIG.S_TDATA_NUM_BYTES [expr $controllerOutWidth / 8] [get_bd_cells $DWCS]
set_property CONFIG.M_TDATA_NUM_BYTES [expr $fulldata / 8] [get_bd_cells $DWCS]

set DWCM [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:1.1 DWCM]
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER] [get_bd_cells $DWCM]
set_property CONFIG.S_TDATA_NUM_BYTES [expr $fulldata / 8] [get_bd_cells $DWCM]
set_property CONFIG.M_TDATA_NUM_BYTES [expr $controllerInWidth / 8] [get_bd_cells $DWCM]

connect_bd_net [get_bd_port $ap_rst_n] [get_bd_pins $out_inv/Op1]
connect_bd_net [get_bd_pins $out_inv/Res] [get_bd_pins $TAP/reset]
connect_bd_net [get_bd_ports $interrupt] [get_bd_pins $TAP/interrupt]
connect_bd_net [get_bd_ports $ap_clk] [get_bd_pins $TAP/clock]
connect_bd_net [get_bd_port $ap_rst_n] [get_bd_pins $DWCM/aresetn]
connect_bd_net [get_bd_port $ap_rst_n] [get_bd_pins $DWCS/aresetn]
connect_bd_net [get_bd_port $ap_clk] [get_bd_pins $DWCM/aclk]
connect_bd_net [get_bd_port $ap_clk] [get_bd_pins $DWCS/aclk]

connect_bd_intf_net [get_bd_intf_ports $S_AXI_LITE] [get_bd_intf_pins $TAP/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_ports $M_AXI] [get_bd_intf_pins $TAP/M_AXI]

connect_bd_intf_net [get_bd_intf_pins $TAP/M_AXIS_CONTROLLER] [get_bd_intf_pins $DWCS/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins $DWCS/M_AXIS] [get_bd_intf_pins $TAP/S_AXIS]

connect_bd_intf_net [get_bd_intf_pins $TAP/S_AXIS_CONTROLLER] [get_bd_intf_pins $DWCM/M_AXIS]
connect_bd_intf_net [get_bd_intf_pins $TAP/M_AXIS] [get_bd_intf_pins $DWCM/S_AXIS]

assign_bd_address
set_property range 64K [get_bd_addr_segs {S_AXI_LITE/SEG_TAP_reg0}]
set_property offset 0x00000000 [get_bd_addr_segs {TAP/M_AXI/SEG_M_AXI_Reg}]
set_property range 4G [get_bd_addr_segs {TAP/M_AXI/SEG_M_AXI_Reg}]
validate_bd_design

set bd_file [get_files ${project_name}.bd]
make_wrapper -files $bd_file -top
add_files -norecurse [file join [file dirname $bd_file] hdl/${project_name}_wrapper.v]
set_property synth_checkpoint_mode Singular $bd_file
generate_target all $bd_file

ipx::package_project -root_dir . -module ${project_name} -generated_files -import_files -force
set_property supported_families {qvirtex7 Beta virtexuplus Beta virtexuplusHBM Beta qzynq Beta zynquplus Beta kintexu Beta versal Beta virtex7 Beta virtexu Beta azynq Beta zynq Beta} [ipx::current_core]
set core [ipx::current_core]
set sim_fg [ipx::get_file_groups xilinx_anylanguagebehavioralsimulation]
foreach file [ipx::get_files -type systemCSource -of_objects $sim_fg] {
      ipx::remove_file [get_property NAME $file] -file_group $sim_fg
}

set_property vendor $vendor $core
set_property library $library $core
set_property display_name "$project_name Free Running Kernel" $core
set_property vendor_display_name $vendor $core
set_property version $version $core

set_property name ap_clk [ipx::get_bus_interfaces CLK.AP_CLK -of_objects $core]
ipx::remove_bus_parameter FREQ_HZ [ipx::get_bus_interfaces ap_clk -of_objects $core]
ipx::remove_bus_parameter PHASE [ipx::get_bus_interfaces ap_clk -of_objects $core]
set_property name ap_rst_n [ipx::get_bus_interfaces RST.AP_RST_N -of_objects $core]
set_property name interrupt [ipx::get_bus_interfaces INTR.INTERRUPT -of_objects $core]

set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core

ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
ipx::check_integrity -quiet $core
ipx::archive_core ./${project_name}_1.0.zip $core

ipx::unload_core component_1
exit

)";

ExecutionResult CreateVivadoProject::executeStep(mlir::ModuleOp *mod) {
  namespace fs = std::filesystem;

  fs::create_directory(config.targetDir);
  fs::create_directory(config.targetDir / config.tmpdir);
  fs::create_directory(config.targetDir / "src");
  fs::path srcPath = config.targetDir / "src";

  // copy source files into target dir
  for (const auto& from : config.sourceFilePaths) {
    try {
      fs::copy(from, srcPath, fs::copy_options::overwrite_existing);
    } catch (const fs::filesystem_error& err) {
      return failure(
        fmt::format("File {} could not be copied to {}: {}", from.string(), srcPath.string(), err.what())
      );
    }
  };

  // write verilog source to target dir
  spdlog::info("Exporting verilog sources to: {}", srcPath.string());
  if (failed(::circt::exportSplitVerilog(*mod, srcPath.string())))
    return failure("exportSplitVerilog() failed");

  // remove filelist.f, rename AXI4StreamMapper.sv, get whole file list
  std::string fileList;

  {
    fs::remove(srcPath / "filelist.f");
    fs::rename(srcPath / "AXI4StreamMapper.sv", srcPath / "AXI4StreamMapper.v");

    for (const fs::directory_entry& entry : fs::directory_iterator(srcPath))
      if (entry.is_regular_file())
        fileList += "src/" + entry.path().filename().string() + " ";
  }

  fs::path tclPath = config.targetDir / "package.tcl";

  // construct package tcl script
  {
    spdlog::info("Generating packaging script in: {}", tclPath.string());
    ControllerDescription *desc = getContext()->get<ControllerDescription>();

    std::string tclSource = fmt::format(TCL_PREAMBLE,
      fmt::arg("projectName", config.projectName),
      fmt::arg("fileList", fileList),
      fmt::arg("liteDataWidth", desc->liteDataWidth),
      fmt::arg("liteAddrWidth", desc->liteAddrWidth),
      fmt::arg("memAddrWidth", desc->memAddrWidth),
      fmt::arg("memDataWidth", desc->memDataWidth),
      fmt::arg("controllerInWidth", desc->sAxisControllerWidth),
      fmt::arg("controllerOutWidth", desc->mAxisControllerWidth),
      fmt::arg("version", config.version)
    ) + TCL_PACKAGE;

    std::ofstream outFile(tclPath);

    if (!outFile.is_open())
      return failure(
        fmt::format("Could not open file {}", tclPath.string())
      );

    outFile << tclSource;
  }

  auto options = getContext()->get<interface::Configuration>();

  // call vivado to create the project
  if (option::launchVivado.get(*options)) {
    auto pwd = std::filesystem::current_path();
    std::filesystem::current_path(config.targetDir);
    spdlog::info("--vivado was specified: Calling vivado...");

    try {
      execShell({
        "vivado", "-mode", "batch", "-source", tclPath.filename().string(), "-nojournal", "-nolog"
      });
    } catch (const std::runtime_error& e) {
      std::filesystem::current_path(pwd);
      return failure(e.what());
    }

    std::filesystem::current_path(pwd);
  } else {
    spdlog::info("--vivado was not specified: Continuing without it...");
  }

  // create the kernel information which we later return
  const KernelInfo *kernelInfo = getContext()->get<KernelInfo>();
  kernel = std::make_unique<Kernel>(
    std::to_string(KERNEL_ID),
    "",
    kernelInfo->queryType, // query_type
    kernelInfo->target, //target
    kernelInfo->batchSize, // batch size
    kernelInfo->numFeatures, // num features
    kernelInfo->bytesPerFeature, // bytes per feature
    kernelInfo->numResults, // num results
    kernelInfo->bytesPerResult, // bytes per result
    "uint32_t" // data type
  );

  if (option::tapascoCompose.get(*options)) {
    spdlog::info("--tapasco-compose was specified: Calling tapasco compose...");
    auto result = tapascoCompose();
    if (failed(result))
      return result;
  } else {
    spdlog::info("--tapasco-compose was not specified: Continuing without it...");
  }

  return success();
}

ExecutionResult CreateVivadoProject::tapascoCompose() {
  namespace fs = std::filesystem;

  auto pwd = std::filesystem::current_path();
  std::filesystem::current_path(config.targetDir);

  // TODO: How is the zip file name determined?
  std::string zipString =
    //config.projectName + "_" + std::regex_replace(config.version, std::regex("\\."), "_") + ".zip";
    config.projectName + "_1.0.zip";
  std::string idString = kernel->fileName();
  std::string deviceString = config.device;
  std::string mhzString = "@" + std::to_string(config.mhz) + "Mhz";

  if (!fs::is_regular_file(fs::path(zipString))) {
    std::filesystem::current_path(pwd);
    return failure(
      fmt::format("file {} is not a valid zip file", zipString)
    );
  }

  try {
    execShell({
      "tapasco", "import", zipString, "as", idString, "-p", deviceString
    });
  } catch (const std::runtime_error& e) {
    std::filesystem::current_path(pwd);
    return failure(e.what());
  }

  try {
    // tapasco compose [spnc x 1] @200Mhz -p ultra96v2
    execShell({
      "tapasco", "compose", "[", config.projectName, "x", "1", "]", mhzString, "-p", deviceString
    });
  } catch (const std::runtime_error& e) {
    std::filesystem::current_path(pwd);
    return failure(e.what());
  }

  std::filesystem::current_path(pwd);

  return success();
}

void CreateVivadoProject::execShell(const std::vector<std::string>& cmd) {
  std::string cmdString;
  for (const std::string& s : cmd)
    cmdString.append(s + " ");
  spdlog::info("Executing shell command: {}", cmdString);
  Command::executeExternalCommand(cmd);
}

}