# MoE active parameters note

Active parameter count is what dominates inference latency on MLX, even
though the model card advertises the total parameter count. For Qwen3
MoE the active path is roughly 3.5B per token — that's the budget you
plan against, not the 35B headline figure.

## Conclusion

When sizing context windows, treat active params as the constraint.
