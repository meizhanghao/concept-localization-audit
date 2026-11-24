# Repository maintenance

- Evaluate proposed changes against the mathematics, executed results and
  manuscript scope before implementing them.
- Keep numerical reproduction self-contained and distinguish it from full model
  training regeneration. Document dependencies and known limitations.
- Validate input hashes, execute the relevant checks, and review scientific
  claims in the documentation before committing. Never commit credentials,
  raw restricted datasets or unnecessary feature caches.
- For this project's author workspace, canonical experiments are maintained in
  `D:\new_cun_ccf\main_python`; the current paper is maintained in
  `D:\new_cun_ccf\lunwen_ccf\BlockSys2027\paper`; this local repository is
  `D:\new_cun_ccf\github_cun\main`.
- Make direct edits within these three directories. Do not create backups or
  alternative working copies elsewhere without asking the author.
- Results incorporated into the paper must have their corresponding code and
  necessary numerical inputs synchronized here. Check manuscript/repository/
  experiment consistency, push the reviewed commit once, then verify remote
  commit agreement. Do not move existing release tags to a different commit.
