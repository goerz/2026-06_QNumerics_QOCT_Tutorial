using CodeTracking: @code_string

# A thin wrapper that renders a Julia code string as a Markdown code block.
#
# We deliberately expose *only* a `text/markdown` representation (plus a plain
# text fallback). Returning a `Markdown.MD` (via `Markdown.parse`) would also
# advertise a `text/latex` representation, in which code blocks become
# `\begin{verbatim}…\end{verbatim}`. VS Code feeds that to KaTeX, which has no
# `verbatim` environment and errors out. Exposing only `text/markdown` makes the
# output render consistently in both JupyterLab and VS Code.
struct CodeBlock
    code::String
end

function Base.show(io::IO, ::MIME"text/markdown", c::CodeBlock)
    # The trailing space after the code works around
    # https://github.com/jupyterlab/jupyterlab/issues/16112
    print(io, "````julia\n", c.code, " \n````")
end

Base.show(io::IO, ::MIME"text/plain", c::CodeBlock) = print(io, c.code)

macro show_code(expr...)
    return quote
        CodeBlock(@code_string $(expr...))
    end
end
