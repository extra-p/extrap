Output formatting documentation
===============================

It is possible to customize the output of Extra-P by using placeholders. Use placeholders enclosed in curly brackets in
any string. You can modify the behaviour of the placeholders, by specifying additional options. Possible placeholders
and their matching options are:

- `{parameters}`
    - `sep`
    - `format`
        - `{parameter}`
- `{points}`
    - `sep`
    - `format`
        - `{point}`
            - `sep`
            - `format`
                - `{coordinate}`
                - `{parameter}`
- `{measurements}`
    - `sep`
    - `format`
        - `{point}`
            - `sep`
            - `format`
                - `{coordinate}`
                - `{parameter}`
        - `{mean}`
        - `{median}`
        - `{std}`
        - `{min}`
        - `{max}`
- `{metric}`
- `{callpath}`
- `{model}`
- `{smape}`
- `{rrss}`
- `{rss}`
- `{re}`
- `{ar2}`

The placeholder name is separated from the options by a colon (like so `{<placholder_name>:<options>}`).
`sep` is used to specify the string used to separate each entry in a list, while `format`
is used to specify how each entry is formatted. `sep` and `format` are separated by a semicolon. Their values have to be
enclosed by apostrophes (`'`). Apostrophes that should occur literally in these values must be escaped with a
backslash (like so `\'`). Curly brackets are reserved for placeholders, if they should appear in the final output they
have to be escaped by doubling them (`{{` or `}}`). Placeholders with numerical values can be formatted based on the
[python format specification mini language](https://docs.python.org/3/library/string.html#format-specification-mini-language)
to specify decimal places and precision. With `{metric}, {callpath}, {points}`
and `{parameters}`, you can remove duplicates by adding `?` before the placeholder names.

**Example formatting strings:**

```
 "{?metric}, {callpath}: errors: {rss:.2e}, {re}"
 "{{{points: sep:' \' ' ; format:'({point})'}}}"
 "{measurements: sep:', '; format:'{point: sep:';';format:'{parameter}{coordinate}'}--{mean}\n'}"
```