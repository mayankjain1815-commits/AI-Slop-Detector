# slop-translator
Viewing AI slop as a translation problem

## Changes I would make if I had to do it again
### Data generation
- Use priority queue for rate limiter
- Follow up if `<POST>` and `</POST>` tags weren't found or if post might contain a `[placeholder]`
  - This could also help address any model weirdness/breakage (e.g. kimi k2 would tend to form recursive loops)
