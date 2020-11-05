File format documentation
=========================

* [Text file format](#text-file-format)
* [TaLPas format](#talpas-format)
* [JSON format](#json-format)
* [JSON Lines format](#json-lines-format)
* [Cube file format](#cube-file-format)
* [Experiment format](#experiment-format)
* [Extra-P 3.0 file format](#extra-p-30-file-format)

Text file format
----------------
The text file format is a basic input format consisting of 5 sections:

- PARAMETER: 
  This section defines the names of the modeling parameters separated by spaces.
  Also, multiple sections can be used. 
  
  Example: `PARAMETER <name> <...>`
  
- POINTS: Defines the coordinates of the measurement points. Uses braces to separate the points consisting of coordinates.

  Example: `POINTS ((<coordinate1>) (<coordinate2>) (<...>)) ((<coordinate1b>) (<coordinate2b>) (<...>)) (<...>)`
  
- METRIC: Defines the metric for the following data sections. Is optional. Applies to all following data sections until another metric is defined.

  Example: `METRIC <name>`

- REGION: Defines the region/callpath for the following data sections.

  Example: `REGION <callpath>`

- DATA: Defines the measurement data for one point. Use one data section for each point. 
  The number of sections after a region section must exactly equal the number of points.
  
  Example: `DATA <value> <...>`

A general subset of the format is defined by the following ABNF grammar:

```ABNF
ExtraPTextFile = *CommentLine 1*ParameterDefinition 
			   1*PointsDefinition 1*MeasurementDefinition
MeasurementDefinition = 1*([MetricDefinition] 1*(RegionDefinition 1*DataDefinition)) 
MeasurementDefinition =/ 1*(RegionDefinition 1*([MetricDefinition] 1*DataDefinition))
    ; Each DataDefinition corresponds to one Point, determined by the order.
    ; Therefore, the number of DataDefinitions per MeasurementDefinition 
    ; must match the number of Points

; Section names
PARAMETER-TAG = %d80.65.82.65.77.69.84.69.82 ; PARAMETER
POINTS = %d80.79.73.78.84.83 ; POINTS
METRIC = %d77.69.84.82.73.67 ; METRIC
REGION = %d82.69.71.73.79.78 ; REGION
DATA = %d68.65.84.65 ; DATA

; Sections
CommentLine = "#" [STRING] 1*NEWLINE

ParameterDefinition = PARAMETER-TAG 1*(SPACE Parameter) LineEnd
Parameter = NAME

PointsDefinition = POINTS 1*(SPACE Point) LineEnd
Point = ("(" Coordinate *(SPACE Coordinate) ")") 
    ; A Point must consist of exactly one Coordinate for each Parameter
    ; If exactly one Parameter is defined the braces of Point are optional
Coordinate = ["("] NUMBER [")"]

MetricDefinition = METRIC SPACE NAMESTRING LineEnd

RegionDefinition = REGION SPACE Callpath LineEnd
Callpath = NAMESTRING 1*("->" NAMESTRING)
    ; Namestring should not contain "->" otherwise it is separated

DataDefinition = DATA 1*(SPACE NUMBER) LineEnd

LineEnd = 1*NEWLINE *CommentLine 

NEWLINE = LF / CRLF
SPACE = WSP
NAME = 1*VCHAR
STRING = *(VCHAR / WSP)
NAMESTRING = NAME STRING
NUMBER = ["-" / "+"] 1*DIGIT ["." *DIGIT]
```

TaLPas format
-------------
The TaLPas format consist of one line for each measurement.
The lines are structured as follows:

```
{"parameters":{"<parameter name>":<parameter value>};"metric":"<metric name>";"callpath":"<callpath>";"value":<measurement value>}
```

JSON format 
-----------

Extra-P supports two different JSON formats (`*.json`).
The newer, recommended one is less verbose. 
The structure is defined by the following schema and is also shown in this example:

```json
{
  "parameters": [
    "<parameter1>",
    "..."
  ],
  "measurements": {
    "<callpath1>": {
      "<metric1>": [
        {
          "point": [
            "<coordinate_value1>",
            "..."
          ],
          "values": [
            "<value1>",
            "..."
          ]
        }
      ],
      "...": []
    },
    "...": {}
  }
}
```

#### Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema",
  "type": "object",
  "title": "Extra-P JSON input file schema",
  "required": [
    "parameters",
    "measurements"
  ],
  "properties": {
    "parameters": {
      "$id": "#/properties/parameters",
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "string"
      }
    },
    "measurements": {
      "$id": "#/properties/measurements",
      "type": "object",
      "minProperties": 1,
      "additionalProperties": {
        "title": "callpath",
        "type": "object",
        "minProperties": 1,
        "additionalProperties": {
          "title": "metric",
          "type": "array",
          "items": {
            "type": "object",
            "required": [
              "point",
              "values"
            ],
            "properties": {
              "point": {
                "type": "array",
                "title": "point",
                "minItems": 1,
                "items": {
                  "type": "number"
                }
              },
              "values": {
                "type": "array",
                "title": "values",
                "minItems": 1,
                "items": {
                  "type": "number"
                }
              }
            }
          }
        }
      }
    }
  }
}
```

The older format is based on referencing parts using ids.
It uses the following structure:

```json
{
    "callpaths": [
        {
            "id": 1,
            "name": "<callpath>"
        }
    ],
    "coordinates": [
        {
            "id": 1,
            "parameter_value_pairs": [
                {
                    "parameter_id": 1,
                    "parameter_value": 0
                }
            ]
        }
    ],
    "measurements": [
        {
            "callpath_id": 1,
            "coordinate_id": 1,
            "id": 1,
            "metric_id": 1,
            "value": 0.0
        }
    ],
    "metrics": [
        {
            "id": 1,
            "name": "<metric name>"
        }
    ],
    "parameters": [
        {
            "id": 1,
            "name": "<parameter name>"
        }
    ]
}
```

For more examples see [tests/data/json](../tests/data/json).

JSON Lines format
-----------------
Extra-P supports the following JSON Lines format (`*.jsonl`) in addition to the JSON formats.
Herby, each line of the file, delimited with '\n', contains a JSON object which describes one measurement value.
The minimal required structure of the JSON objects is as follows:

```json
{ "params": { "<parameter1>": 0, "...": "..." }, "value": 0.0 }
```

Value may also be a list of values:

```json
{ "...": "...", "value": [0.0, 0.1, 0.2] }
```

Optionally the callpath and/or metric can also be defined:

```json
{ "...": "...", "callpath": "<callpath>", "metric": "<metric>"}
```

For more examples see [tests/data/jsonlines](../tests/data/jsonlines).

Cube file format
-----------------------
The Cube file format is based on a directory structure. 
All measurements are organized in a directory, which contains directories for each measurement point.
Each directory for a measurement point must contain one or more Cube files (*.cubex), containing the actual measurement.
The names of Cube files must not start with a dot `.`, otherwise they will be ignored.

The name of the dictionary indicates the measurement point for the different parameters.
It should be structured in the following way:

```
NAME = [PREFIX "."] PARAMETER-VALUE-PAIRS [".r" REPETITION-NUMBER]
PARAMETER-VALUE-PAIRS = PARAMETER-NAME PARAMETER-VALUE *(["."/","] PARAMETER-NAME PARAMETER-VALUE) 
```

Examples for possible name structures are:

* `mm.a1.1b1.1c1.1`
* `mm.x1y1z1`
* `mm.x1y1z1.r1`
* `mm.a1,1.b1,1.c1,1.r1`
* `mm.x1.1,y1,1,z1.1.r1`
* `mm.x1.1.y1.1.z1.1.r1`
* `mm.x1.y1.z1.r1`
* `x1y1z1`

The overall directory structure should be similar to the following:

```
cube file folder
|
+--+mm.x1000y1z1.r1
|  +--profile.cubex
+--+mm.x100y1z1.r1
|  +--profile.cubex
+--+mm.x10y1z1.r1
|  +--profile.cubex
+--+mm.x1y1z1.r1
|  +--profile.cubex
+--+mm.x2000y1z1.r1
|  +--profile.cubex
+--+mm.x250y1z1.r1
|  +--profile.cubex
+--+mm.x25y1z1.r1
|  +--profile.cubex
+--+mm.x500y1z1.r1
|  +--profile.cubex
+--+mm.x50y1z1.r1
   +--profile.cubex
```

Experiment format
-----------------
The Extra-P experiment format (`.extra-p`) is a ZIP-file containing a file called `experiment.json`.
This file contains the actual experiment data serialized to JSON. 
It is generated by serializing an `Experiment` object which represents an experiment internally. 
The basic structure of the `experiment.json` file is given by the following JSON schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "definitions": {
    "Number": {
      "type": ["number", "string"],
      "pattern": "^-?inf$|^nan$|^\\d*\\/\\d*$"
    },
    "MeasurementSchema": {
      "type": "object",
      "properties": {
        "coordinate": {
          "type": "array",
          "items": {"$ref": "#/definitions/Number"}
        },
        "maximum": {"$ref": "#/definitions/Number"},
        "mean": {"$ref": "#/definitions/Number"},
        "median": {"$ref": "#/definitions/Number"},
        "minimum": {"$ref": "#/definitions/Number"},
        "std": {"$ref": "#/definitions/Number"}
      },
      "additionalProperties": false
    },
    "ModelSchema": {
      "type": "object",
      "properties": {
        "hypothesis": {
          "type": "object",
          "$ref": "#/definitions/HypothesisSchema"
        }
      },
      "additionalProperties": false
    },
    "HypothesisSchema": {
      "type": "object",
      "properties": {
        "AR2": {"$ref": "#/definitions/Number"},
        "RE": {"$ref": "#/definitions/Number"},
        "RSS": {"$ref": "#/definitions/Number"},
        "rRSS": {"$ref": "#/definitions/Number"},
        "SMAPE": {"$ref": "#/definitions/Number"},
        "_costs_are_calculated": {
          "title": "_costs_are_calculated",
          "type": "boolean"
        },
        "_use_median": {
          "title": "_use_median",
          "type": "boolean"
        },
        "function": {
          "type": "object",
          "$ref": "#/definitions/FunctionSchema"
        }
      },
      "additionalProperties": true
    },
    "FunctionSchema": {
      "type": "object",
      "properties": {
        "compound_terms": {
          "title": "compound_terms",
          "type": "array",
          "items": {
            "type": "object",
            "$ref": "#/definitions/CompoundTermSchema"
          }
        },
        "constant_coefficient": {"$ref": "#/definitions/Number"}
      },
      "additionalProperties": true
    },
    "CompoundTermSchema": {
      "type": "object",
      "properties": {
        "coefficient": {"$ref": "#/definitions/Number"},
        "simple_terms": {
          "title": "simple_terms",
          "type": "array",
          "items": {
            "type": "object",
            "$ref": "#/definitions/SimpleTermSchema"
          }
        }
      },
      "additionalProperties": true
    },
    "SimpleTermSchema": {
      "type": "object",
      "properties": {
        "coefficient": {"$ref": "#/definitions/Number"},
        "exponent": {"$ref": "#/definitions/Number"},
        "term_type": {
          "title": "term_type",
          "type": "string",
          "enum": ["polynomial", "logarithm"]
        }
      },
      "additionalProperties": false
    },
    "ModelGeneratorSchema": {
      "type": "object",
      "properties": {
        "modeler": {
          "type": "object",
          "$ref": "#/definitions/ModelerSchema"
        },
        "models": {
          "type": "object",
          "additionalProperties": {
            "type": "object",
            "additionalProperties": {
              "type": "object",
              "$ref": "#/definitions/ModelSchema"
            }
          }
        },
        "name": {
          "title": "name",
          "type": "string"
        }
      },
      "additionalProperties": false
    },
    "ModelerSchema": {
      "type": "object",
      "properties": {
        "use_median": {
          "title": "use_median",
          "type": "boolean"
        }
      },
      "additionalProperties": true
    },
    "ExperimentSchema": {
      "type": "object",
      "required": [
        "Extra-P"
      ],
      "properties": {
        "Extra-P": {
          "title": "version",
          "type": "string",
          "default": "4.0"
        },
        "measurements": {
          "type": "object",
          "title": "callpaths",
          "additionalProperties": {
            "type": "object",
            "title": "metrics",
            "additionalProperties": {
              "title": "measurements",
              "type": "array",
              "items": {
                "type": "object",
                "$ref": "#/definitions/MeasurementSchema"
              }
            }
          }
        },
        "modelers": {
          "title": "modelers",
          "type": "array",
          "default": [],
          "items": {
            "type": "object",
            "$ref": "#/definitions/ModelGeneratorSchema"
          }
        },
        "parameters": {
          "title": "parameters",
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "scaling": {
          "title": "scaling",
          "type": ["string", "null"],
          "enum": ["strong", "weak", null]
        }
      },
      "additionalProperties": false
    }
  },
  "$ref": "#/definitions/ExperimentSchema"
}
```

Extra-P 3.0 file format
-----------------------
The Extra-P 3.0 file format uses binary encoding of experiments. 
It is only included for backwards compatibility and should not be used for new data.  
