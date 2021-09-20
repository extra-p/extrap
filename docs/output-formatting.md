Output formatting documentation
==========================

It is possible to customize the program output by using placeholders. 
Use placeholders in any string enclosed in curly brackets. Possible placeholders
include:


- {parameters}
	- sep
	- format
		- {parameter}
- {points}
	- sep
	- format
		- {point}
			- sep
			- format
				- {coordinate}
				- {parameter}
- {measurements}
	- sep
	- format
		- {point}
			- sep
			- format
				- {coordinate}
				- {parameter}
		- {mean}
		- {median}
		- {std}
		- {min}
		- {max}
- {metric}
- {callpath}
- {model} 
- {smape}
- {rrss}
- {rss}
- {re}
- {ar2}

sep is used to specify the string used to separate each entry, while format 
is used to specify how each entry is formatted. sep and format are separated 
by a semicolon. Single apostrophes and curly brackets have to be escaped to 
appear in the final output. Placeholders with numerical values can be formatted
to specify decimal places. With `{metric}, {callpath}, {points}` and `{parameters}`, 
you can remove duplicates by adding '?' before placeholder names. 

Example strings:
````
 "{?metric}, {callpath}: errors: {rss:.2e}, {re}"
 "{{{points: sep:' \' ' ; format:'({point})'}}}"
 "{measurements: sep:', '; format:'{point: sep:';';format:'{parameter}{coordinate}'}--{mean}\n'}"
````