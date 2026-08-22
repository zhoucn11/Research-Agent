from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re


SKILLS_ROOT = Path(__file__).resolve().parents[2] / ".agent" / "skills"
SYNTHESIS_SKILL_NAMES = ("literature-review-writing",)


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    instructions: str
    path: Path


def _parse_skill(path: Path) -> SkillDefinition | None:
    skill_file = path / "SKILL.md"
    if not skill_file.is_file():
        return None
    content = skill_file.read_text(encoding="utf-8").strip()
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, flags=re.DOTALL)
    if not match:
        return None
    frontmatter, instructions = match.groups()
    name_match = re.search(r"(?m)^name:\s*(.+?)\s*$", frontmatter)
    description_match = re.search(r"(?m)^description:\s*(.+?)\s*$", frontmatter)
    if not name_match or not description_match:
        return None
    name = name_match.group(1).strip().strip('"\'')
    description = description_match.group(1).strip().strip('"\'')
    if name != path.name or not re.fullmatch(r"[a-z0-9-]{1,64}", name):
        return None
    return SkillDefinition(name, description, instructions.strip(), path)


@lru_cache(maxsize=1)
def discover_skills() -> dict[str, SkillDefinition]:
    if not SKILLS_ROOT.is_dir():
        return {}
    skills = {}
    for path in sorted(SKILLS_ROOT.iterdir()):
        skill = _parse_skill(path) if path.is_dir() else None
        if skill:
            skills[skill.name] = skill
    return skills


def build_synthesis_skill_context(user_text: str, paper_count: int = 0) -> tuple[list[str], str]:
    """仅在综述类写作任务中向 Synthesizer 暴露审核通过的 Skill。"""
    text = str(user_text or "").casefold()
    explicit_review_intent = re.search(
        r"综述|相关工作|文献回顾|总结|综合|归纳|梳理|对比|比较|共同点|差异|研究空白|"
        r"literature review|related work|survey|summari[sz]e|comparison|compare",
        text,
    )
    generic_writing_intent = re.search(r"整理成|写成|整合|形成.*(?:段落|小节|报告)", text)
    if not explicit_review_intent and not (paper_count > 1 and generic_writing_intent):
        return [], ""

    skills = discover_skills()
    names = [name for name in SYNTHESIS_SKILL_NAMES if name in skills]
    if not names:
        return [], ""
    sections = [
        f"### {name}\n{skills[name].instructions}"
        for name in names
    ]
    return names, (
        "\n【Synthesizer 专用写作 Skill】\n"
        "以下规则只约束证据组织与综述写作，不得触发检索、补论文或修改证据。\n"
        + "\n\n".join(sections)
    )
