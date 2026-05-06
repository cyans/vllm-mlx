# Knowledge distillation reading note

I think distillation only works when the teacher's logits actually carry
information the student cannot learn from the labels alone. The classic
soft-target story is well-trodden but the recent line of work on
data-mixture distillation looks more promising for our use case.

## Open questions

- Does response distillation matter for instruction-tuned models?
- How much teacher diversity is enough?
