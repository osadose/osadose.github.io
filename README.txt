Hi there, I'm Ose



project:
  type: book
  output-dir: docs

book:
  title: "Python Git Course"
  author: "Osemwingie Osadolor"
  date: "2025-05-21"
  navbar: 
    logo: slides/template/logo.png  # Use forward slashes for cross-platform compatibility
    title: false
  chapters:
    - index.qmd

    - part: "Week 0"
      chapters:
        - course-materials/week-0/software_list.md
        - course-materials/week-0/vscode_extensions.md

    - part: "Week 1"
      chapters:
        - course-materials/week-1/index.qmd

    - part: "Week 2"
      chapters:
        - course-materials/week-2/git_cheatsheet.md
        - course-materials/week-2/week_2_newsletter.md

    - part: "Week 3"
      chapters:
        - course-materials/week-3/week_3_newsletter.md

    - part: "Week 4"
      chapters:
        - course-materials/week-4/week_4_newsletter.md

    - part: "Week 5"
      chapters:
        - course-materials/week-5/data_cleaning_demo.ipynb

    - part: "Week 6"
      chapters:
        - course-materials/week-6/index.qmd

    - part: "Week 7"
      chapters:
        - course-materials/week-7/background_and_examples.ipynb
        - course-materials/week-7/functions_demo.ipynb
        - course-materials/week-7/v1_importing_custom_functions.ipynb
        - course-materials/week-7/v2_importing_custom_functions.ipynb

    - part: "Week 8"
      chapters:
        - course-materials/week-8/grading_example.ipynb
        - course-materials/week-8/polio_project.ipynb

    - part: "Appendix"
      chapters: 
        - appendix.qmd

format:
  html:
    theme: cosmo
    toc: true
    number-sections: true

editor: visual











