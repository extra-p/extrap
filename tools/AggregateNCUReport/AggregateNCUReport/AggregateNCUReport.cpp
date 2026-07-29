#include "AggregateNCUReport.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <variant>
#include <optional>
#include <unordered_map>
#include <filesystem>
#include <SQLiteCpp/SQLiteCpp.h>
#include <kaitai/kaitaistream.h>
#include "nsight_cuprof_report.h"
namespace fs = std::filesystem;

struct empty_value {
	friend constexpr empty_value operator+(empty_value val, empty_value e) {
		return val;
	}
	template<typename T>
	friend constexpr T operator+(T val, empty_value e) {
		return val;
	}
	template<typename T>
	friend constexpr T operator+(empty_value e, T val) {
		return val;
	}
	friend constexpr bool operator==(empty_value val, empty_value e) {
		return true;
	}
	template<typename T>
	friend constexpr bool operator==(T val, empty_value e) {
		return false;
	}
	template<typename T>
	friend constexpr bool operator==(empty_value e, T val) {
		return false;
	}

};


namespace std {
	std::string to_string(empty_value val) {
		return std::string();
	}
}

using value_type = std::variant<empty_value, uint32_t, uint64_t, float, double>;

value_type _convert_metric_value(const ProfileMetricValue& mv) {
	if (mv.has_doublevalue()) {
		return mv.doublevalue();
	}
	if (mv.has_floatvalue()) {
		return mv.floatvalue();
	}
	if (mv.has_uint64value()) {
		return mv.uint64value();
	}
	if (mv.has_uint32value()) {
		return mv.uint32value();
	}
	return empty_value{};
}


struct tuple_hash
{
	using key = std::tuple<unsigned int, unsigned int>;

	size_t operator()(const key& tuple) const
	{
		return std::hash<std::tuple_element_t<0, key>>{}(std::get<0>(tuple))
			+ std::hash<std::tuple_element_t<1, key>>{}(std::get<1>(tuple));
		//+ std::hash<std::tuple_element_t<2, key>>{}(std::get<2>(tuple));
	}
};

struct tuple_hash_string
{
	using key = std::tuple<std::string, unsigned int>;

	size_t operator()(const key& tuple) const
	{
		return std::hash<std::tuple_element_t<0, key>>{}(std::get<0>(tuple))
			+ std::hash<std::tuple_element_t<1, key>>{}(std::get<1>(tuple));
		//+std::hash<std::tuple_element_t<2, key>>{}(std::get<2>(tuple));
	}
};

unsigned int _get_kernel_id(std::unordered_map<std::string, unsigned int>& kernel_names, const std::string& kernel_name)
{
	unsigned int kernel_id;
	auto search = kernel_names.find(kernel_name);
	if (search != kernel_names.end()) {
		kernel_id = search->second;
	}
	else {
		kernel_id = kernel_names.size();
		kernel_names[kernel_name] = kernel_id;
	}
	return kernel_id;
}

template<typename K, typename T>
void aggregate_values(const int num_files, const int fidx, const K& key, const value_type& value, T& aggregated_values)
{
	auto search = aggregated_values.find(key);
	if (search != aggregated_values.end()) {
		if (std::holds_alternative<empty_value>(search->second[fidx])) {
			aggregated_values[key][fidx] = value;
		}
		else {

			aggregated_values[key][fidx] = std::visit([](auto&& a, auto&& b) -> value_type {
				return a + b;
				}, search->second[fidx], value);
		}
	}
	else
	{
		auto value_list = std::vector<value_type>(num_files);
		value_list[fidx] = value;
		aggregated_values[key] = value_list;
	}
}

void try_load_database(std::filesystem::path& path, std::unique_ptr<SQLite::Database>& db, std::unique_ptr<SQLite::Statement>& kernel_info)
{
	fs::path db_path = path;
	db_path.replace_extension(".sqlite");
	if (fs::exists(db_path)) {
		std::cout << "Found Database" << db_path << std::endl;
		db = std::make_unique<SQLite::Database>(db_path.string());
		if (!db->tableExists("EXTRAP_RESOLVED_CALLPATHS")) {
			db.reset();
		}
		kernel_info = std::make_unique<SQLite::Statement>(*db, "WITH cupti_kernel AS (																					 \
				SELECT correlationId,																							 \
				gridId,																											 \
				(end - start)                                            AS durationGPU,										 \
				shortName,																										 \
				('(' || gridX || ',' || gridY || ',' || gridZ || ')')    AS grid,												 \
				('(' || blockX || ',' || blockY || ',' || blockZ || ')') AS block,												 \
				sharedMemoryExecuted																							 \
				FROM CUPTI_ACTIVITY_KIND_KERNEL																					 \
			),																													 \
				cupti_activity AS(																								 \
					SELECT correlationId,																						 \
					gridId,																										 \
					value AS demangledName,																						 \
					grid,																										 \
					block,																										 \
					sharedMemoryExecuted,																						 \
					durationGPU																									 \
					FROM cupti_kernel AS CAKK																					 \
					LEFT JOIN StringIds ON shortName = StringIds.id																 \
					WHERE correlationId IS NOT NULL																				 \
				)																												 \
				SELECT demangledName AS name,																					 \
				grid,																											 \
				block,																											 \
				sharedMemoryExecuted,																							 \
				callpath																										 \
				FROM cupti_activity																								 \
				LEFT JOIN EXTRAP_RESOLVED_CALLPATHS ON EXTRAP_RESOLVED_CALLPATHS.correlationId = CUPTI_ACTIVITY.correlationId	 \
				");
	}
}

int process_files(std::vector<fs::path>& paths)
{
	int num_files = paths.size();

	std::vector<std::unique_ptr<BlockHeader>> string_tables;
	std::unordered_map<std::tuple<unsigned int, unsigned int>, std::vector<value_type>, tuple_hash> aggregated_values_per_kernel;
	std::unordered_map<std::tuple<std::string, unsigned int>, std::vector<value_type>, tuple_hash_string> aggregated_values_per_call;
	std::unordered_map<std::string, unsigned int> kernel_names;

	for (int fidx = 0; fidx < num_files; fidx++) {

		std::unique_ptr<SQLite::Database> db;
		std::unique_ptr<SQLite::Statement> kernel_info;
		bool kernel_info_has_line = true;
		try_load_database(paths[fidx], db, kernel_info);
		std::cout << "Reading " << paths[fidx] << std::endl;
		std::ifstream is(paths[fidx], std::ifstream::binary);
		kaitai::kstream ks(&is);
		nsight_cuprof_report_t report(&ks);

		uint64_t str_tab_ctr = 0;
		for (auto& block : *report.blocks()) {
			if (!block->header()->data()->stringtable().strings().empty()) {
				if (fidx == 0) {
					string_tables.emplace_back(block->header()->retrieve_data_ptr());
				}
				else {
					if (block->header()->data()->stringtable().strings().size() != string_tables[str_tab_ctr]->stringtable().strings().size()) {
						std::cout << "String tables do not match." << std::endl;
						return 1;
					}
					size_t size = block->header()->data()->stringtable().strings().size();
					for (int i = 0; i < size; i++) {
						if (block->header()->data()->stringtable().strings()[i] != string_tables[str_tab_ctr]->stringtable().strings()[i]) {
							std::cout << "String tables do not match: " << i << " " << block->header()->data()->stringtable().strings()[i] << " : " << string_tables[0]->stringtable().strings()[i] << std::endl;
							return 1;
						}
					}
				}
				str_tab_ctr++;
			}
			if (block->payload()->num_sources() > 0) {

			}
			if (block->payload()->num_results() > 0) {
				for (auto& results_raw : *block->payload()->results()) {
					auto& result = *results_raw->entry()->data();
					auto kernel_id = _get_kernel_id(kernel_names, result.kerneldemangledname());
					assert(kernel_id < kernel_names.size());
					if (kernel_info != nullptr) {
						if (kernel_info_has_line) {
							kernel_info_has_line = kernel_info->executeStep();
						}
						if (!kernel_info_has_line) {
							auto& name = string_tables[0]->stringtable().strings()[kernel_id];
							std::cout << "No further kernels in database: id: " << kernel_id << " name: " << name << std::endl;
							//return 3;
							for (auto& mv : result.metricresults())
							{
								auto value = _convert_metric_value(mv.metricvalue());
								if (!std::holds_alternative<empty_value>(value)) {
									auto key = std::make_tuple(kernel_id, mv.nameid());
									aggregate_values(num_files, fidx, key, value, aggregated_values_per_kernel);

									auto key2 = std::make_tuple("UNMATCHED->" + name + "->GPU " + name, mv.nameid());
									aggregate_values(num_files, fidx, key2, value, aggregated_values_per_call);
								}
							}
						}
						else {
							auto name = kernel_info->getColumn("name").getString();
							auto callpath = kernel_info->getColumn("callpath").getString();
							if (result.kernelfunctionname() != name) {
								std::cout << name << "does not match" << result.kernelfunctionname() << std::endl;
								return 2;
							}
							for (auto& mv : result.metricresults())
							{
								auto value = _convert_metric_value(mv.metricvalue());
								if (!std::holds_alternative<empty_value>(value)) {
									auto key = std::make_tuple(kernel_id, mv.nameid());
									aggregate_values(num_files, fidx, key, value, aggregated_values_per_kernel);
									auto key2 = std::make_tuple(callpath + "->" + name + "->GPU " + name, mv.nameid());
									aggregate_values(num_files, fidx, key2, value, aggregated_values_per_call);
								}
							}
						}
					}
					else {
						for (auto& mv : result.metricresults())
						{
							auto value = _convert_metric_value(mv.metricvalue());
							if (!std::holds_alternative<empty_value>(value)) {
								auto key = std::make_tuple(kernel_id, mv.nameid());
								aggregate_values(num_files, fidx, key, value, aggregated_values_per_kernel);
							}
						}
					}
				}
			}
		}
	}
	char US = '\x1F';
	char RS = '\x1E';
	fs::path outpath = paths[0];
	outpath += ".agg";
	std::cout << "Writing " << outpath << std::endl;
	std::ofstream outs(outpath, std::ifstream::binary);
	outs << "Extra-P NCU Aggregation 1.0\n";
	outs << "!BEGIN Info\n";
	outs << "count" << RS << num_files << '\n';
	outs << "!END Info\n";
	outs << "!BEGIN Strings\n";

	for (auto& table : string_tables) {
		for (auto& string : table->stringtable().strings()) {
			outs << string << "\n";
		}
		if (string_tables.size() > 1) {
			outs << "!SEPARATE Strings\n";
		}
	}
	outs << "!END Strings\n";
	outs << "!BEGIN Kernels\n";
	for (auto& kv : kernel_names) {
		outs << kv.second << RS << kv.first << "\n";
	}
	outs << "!END Kernels\n";
	outs << "!BEGIN Aggregated Measurements Kernel\n";
	for (auto& [key, values] : aggregated_values_per_kernel) {
		outs << std::get<0>(key) << RS << std::get<1>(key) << RS;
		for (auto& value : values) {
			if (!std::holds_alternative<empty_value>(value)) {
				outs << std::visit([](auto&& a) -> std::string { if (a == 0) { return "0"; } return std::to_string(a); }, value) << US;
			}
		}
		outs.seekp(-1, std::ios_base::cur);
		outs << '\n';
	}
	outs << "!END Aggregated Measurements Kernel\n";
	outs << "!BEGIN Aggregated Measurements Callpath\n";
	for (auto& [key, values] : aggregated_values_per_call) {
		outs << std::get<0>(key) << RS << std::get<1>(key) << RS;
		for (auto& value : values) {
			if (!std::holds_alternative<empty_value>(value)) {
				outs << std::visit([](auto&& a) -> std::string { if (a == 0) { return "0"; } return std::to_string(a); }, value) << US;
			}
		}
		outs.seekp(-1, std::ios_base::cur);
		outs << '\n';
	}
	outs << "!END Aggregated Measurements Callpath\n";
	std::cout << "Finished " << outpath << std::endl << std::endl;
	return 0;
}



int main(int argc, char* argv[])
{
	if (argc < 2) {
		std::cout << "Missing Path." << std::endl;
		return 1;
	}
	std::vector<fs::path> paths;
	for (int i = 1; i < argc; i++) {
		if (fs::exists(argv[i])) {
			paths.emplace_back(argv[i]);
		}
		else {
			std::cout << "Path does not exist: " << argv[i] << " : Skipped." << std::endl;
		}
	}
	if (paths.empty()) {
		std::cout << "No valid paths given. Exiting." << std::endl;
		return 2;
	}
	if (std::all_of(paths.cbegin(), paths.cend(), [](auto& path) {
		return !fs::is_directory(path);
		})) {
		int retval = process_files(paths);
		return retval;
	}
	else
	{
		for (auto& path : paths) {
			if (fs::is_directory(path)) {
				std::vector<fs::path> file_paths;
				for (auto const& dir_entry : fs::directory_iterator(path)) {
					if (dir_entry.is_directory()) {
						std::vector<fs::path> file_paths;
						for (auto const& dir_entry : fs::directory_iterator(dir_entry)) {
							if (!dir_entry.is_directory() && dir_entry.path().extension() == ".ncu-rep") {
								file_paths.emplace_back(dir_entry.path());
							}
						}
						if (!file_paths.empty()) {
							int retval = process_files(file_paths);
							if (retval != 0) {
								return retval;
							}
						}
					}
					else {
						if (dir_entry.path().extension() == ".ncu-rep") {
							file_paths.emplace_back(dir_entry.path());
						}
					}
				}
				if (!file_paths.empty()) {
					int retval = process_files(file_paths);
					if (retval != 0) {
						return retval;
					}
				}
			};
		}
	}



	return 0;
}
