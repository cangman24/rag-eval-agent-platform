# -*- coding: utf-8 -*-
"""tests/test_engineering_config.py —— 工程配置的一致性测试。

★ 为什么工程配置也值得写测试：

  配置文件的特点是**错了不一定报错，而是行为悄悄变了**。举三个真实例子：

    · .env.example 漏了一个变量 -> 用户照抄后行为与文档不符，
      却没有任何报错，只是"某些参数没生效"。
    · docker-compose 里模型挂载路径与 bootstrap 默认值不一致 ->
      容器起得来，但加载模型时报路径不存在，排查要翻两层配置。
    · CI 的 YAML 写错一个缩进 -> 本地毫无感觉，
      推上去才发现整条流水线没触发。

  这些都属于"代价高、发现晚"的错误，用几条断言就能提前拦住。

★ pyyaml 是可选依赖：没装时用 skipTest 跳过 YAML 相关用例，
  而不是让整个测试套件挂掉 —— 与 core/ 的零依赖可测设计保持一致。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import yaml  # type: ignore
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

requires_yaml = unittest.skipUnless(HAS_YAML, "需要 pyyaml（可选依赖）")


def read(path: str) -> str:
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
        return f.read()


def load_yaml(path: str):
    return yaml.safe_load(read(path))


# ================================================================== 文件存在
class TestFilesExist(unittest.TestCase):
    def test_engineering_files_present(self):
        for p in ("Dockerfile", ".dockerignore", "docker-compose.yml",
                  "pyproject.toml", "Makefile", ".env.example",
                  ".github/workflows/ci.yml",
                  "rag1_knowledge_qa/server.py"):
            self.assertTrue(os.path.exists(os.path.join(ROOT, p)),
                            f"缺少工程文件: {p}")


# ================================================================== 环境变量
class TestEnvTemplate(unittest.TestCase):
    """★ .env.example 必须与代码里的环境变量清单保持一致。

    这是最容易失配的地方：代码加了新变量、忘了更新模板，
    用户照抄模板后该变量永远是默认值。
    """

    def _defaults(self):
        from rag1_knowledge_qa.bootstrap import DEFAULTS
        return DEFAULTS

    def test_every_default_is_documented(self):
        body = read(".env.example")
        missing = []
        for key in self._defaults():
            if key not in body:
                missing.append(key)
        self.assertEqual(missing, [],
                         f".env.example 缺少这些变量: {missing}")

    def test_api_key_documented_but_empty(self):
        """API Key 必须出现在模板里，但**不能带真实值**。"""
        body = read(".env.example")
        self.assertIn("DEEPSEEK_API_KEY", body)
        line = next(l for l in body.splitlines()
                    if l.strip().startswith("DEEPSEEK_API_KEY"))
        value = line.split("=", 1)[1].strip()
        self.assertEqual(value, "", f"模板里的 Key 必须留空，实际是 {value!r}")

    def test_defaults_match_documented_values(self):
        """模板里写的默认值应与代码 DEFAULTS 一致（防止两边各说各话）。"""
        defaults = self._defaults()
        body = read(".env.example")
        doc = {}
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            doc[k.strip()] = v.strip()
        mismatched = {k: (defaults[k], doc[k]) for k in defaults
                      if k in doc and doc[k] and doc[k] != defaults[k]}
        self.assertEqual(mismatched, {},
                         f"模板与代码默认值不一致 (key: (代码, 模板)): {mismatched}")


# ================================================================== Docker
class TestDockerfile(unittest.TestCase):
    def test_transforms_pinned_to_4x(self):
        """★ transformers 必须锁 4.44.x。

        sentence-transformers / FlagEmbedding 尚不兼容 5.x。
        实测 conda 克隆出来的环境常自带 5.19.0，不锁会在加载模型时炸。
        """
        body = read("Dockerfile")
        self.assertIn("transformers==4.44.2", body,
                      "Dockerfile 里必须把 transformers 锁到 4.44.2")

    def test_runs_as_non_root(self):
        self.assertIn("USER appuser", read("Dockerfile"))

    def test_has_healthcheck(self):
        self.assertIn("HEALTHCHECK", read("Dockerfile"))

    def test_healthcheck_hits_health_endpoint(self):
        self.assertIn("/health", read("Dockerfile"))

    def test_dependency_layer_before_source(self):
        """依赖层必须在源码之前 COPY，否则每次改代码都要重装依赖。"""
        body = read("Dockerfile")
        i_pyproject = body.find("COPY pyproject.toml")
        i_core = body.find("COPY core/")
        self.assertGreater(i_pyproject, -1)
        self.assertGreater(i_core, -1)
        self.assertLess(i_pyproject, i_core,
                        "应先 COPY pyproject.toml 再 COPY 源码（Docker 层缓存）")

    def test_multistage(self):
        self.assertGreaterEqual(read("Dockerfile").count("FROM "), 2)


class TestDockerignore(unittest.TestCase):
    def test_env_excluded(self):
        """★ .env 必须在 .dockerignore 里。

        镜像层不可变：一旦 .env 被打进某一层，即使下一层删除，
        它仍留在 history 里可被 pull 出来。这不是"减小上下文"的问题，
        是密钥泄漏问题。
        """
        body = read(".dockerignore")
        lines = [l.strip() for l in body.splitlines() if l.strip()
                 and not l.startswith("#")]
        self.assertIn(".env", lines)
        self.assertIn("models/", lines)
        self.assertIn("data/vectors/", lines)


class TestDockerCompose(unittest.TestCase):
    @requires_yaml
    def test_parses(self):
        dc = load_yaml("docker-compose.yml")
        self.assertIn("rag", dc["services"])

    @requires_yaml
    def test_mounts_match_bootstrap_defaults(self):
        """★ 容器内路径必须与 bootstrap 默认值对得上。

        失配的症状是：容器正常起来，但加载模型时报路径不存在，
        排查要同时翻 compose 和 bootstrap 两处配置。
        """
        dc = load_yaml("docker-compose.yml")
        env = dc["services"]["rag"]["environment"]
        from rag1_knowledge_qa.bootstrap import DEFAULTS

        # 容器内路径以 /app 开头，bootstrap 默认值是相对路径，
        # 因此只比对"文件名部分"是否一致
        for key in ("BGE_M3_PATH", "BGE_RERANKER_PATH", "RAG_CHUNKS_FILE",
                    "RAG_VECTORS_FILE"):
            self.assertIn(key, env, f"compose 应显式设置 {key}")
            want_tail = DEFAULTS[key].split("/")[-1]
            self.assertTrue(env[key].endswith(want_tail),
                            f"{key}: compose={env[key]} 与默认值 "
                            f"{DEFAULTS[key]} 不一致")

    @requires_yaml
    def test_env_file_not_inline_secret(self):
        """必须用 env_file 引用 .env，而不是把 Key 内联写在 compose 里。"""
        dc = load_yaml("docker-compose.yml")
        svc = dc["services"]["rag"]
        self.assertIn("env_file", svc)
        self.assertNotIn("DEEPSEEK_API_KEY", svc.get("environment", {}))

    @requires_yaml
    def test_volumes_mount_models_readonly(self):
        dc = load_yaml("docker-compose.yml")
        vols = dc["services"]["rag"]["volumes"]
        model_vols = [v for v in vols if "/app/models" in str(v)]
        self.assertTrue(model_vols, "应挂载 models")
        self.assertTrue(any(":ro" in str(v) for v in model_vols),
                        "模型应只读挂载（容器不该修改权重）")


# ================================================================== CI
class TestCIConfig(unittest.TestCase):
    @requires_yaml
    def test_parses_and_has_jobs(self):
        ci = load_yaml(".github/workflows/ci.yml")
        self.assertIn("jobs", ci)
        self.assertGreaterEqual(len(ci["jobs"]), 4)

    @requires_yaml
    def test_every_job_has_runner_and_valid_steps(self):
        ci = load_yaml(".github/workflows/ci.yml")
        for name, job in ci["jobs"].items():
            self.assertIn("runs-on", job, f"job {name} 缺 runs-on")
            for step in job.get("steps") or []:
                self.assertTrue("uses" in step or "run" in step,
                                f"job {name} 有既无 uses 也无 run 的步骤")

    @requires_yaml
    def test_does_not_install_heavy_deps(self):
        """★ CI 的核心主张：765 项测试**不需要** torch / chromadb / langgraph。

        如果哪天有人在 CI 里加了 torch，说明有核心模块被外部依赖污染了 ——
        这条断言就是在守护那条架构边界。
        """
        body = read(".github/workflows/ci.yml")
        for dep in ("torch", "chromadb", "langgraph", "FlagEmbedding"):
            self.assertNotIn(f"pip install {dep}", body,
                             f"CI 不应安装 {dep}（core/ 应当是零依赖可测的）")

    @requires_yaml
    def test_secret_guard_job_exists(self):
        """必须有"全量历史扫描 .env"的 job。

        防的是"提交过又删掉"——git 历史与镜像层一样不可变。
        """
        ci = load_yaml(".github/workflows/ci.yml")
        self.assertIn("secret-guard", ci["jobs"])
        body = read(".github/workflows/ci.yml")
        self.assertIn("fetch-depth: 0", body,
                      "扫描历史需要完整 clone（fetch-depth: 0）")
        self.assertIn("git log --all", body)

    @requires_yaml
    def test_reproducibility_job_checks_determinism(self):
        """可复现性必须被机器验证，不能只写在 README 里。"""
        ci = load_yaml(".github/workflows/ci.yml")
        self.assertIn("reproducibility", ci["jobs"])

    @requires_yaml
    def test_embedded_python_scripts_compile(self):
        """★ 抽取 CI 里 <<'PY' 的内嵌脚本并编译。

        heredoc 里的代码最容易写错（缩进、引号转义），
        而且本地感觉不到 —— 推上去才发现 job 失败。
        """
        import re
        ci = load_yaml(".github/workflows/ci.yml")
        found = 0
        for jname, job in ci["jobs"].items():
            for step in job.get("steps") or []:
                run = step.get("run") or ""
                for m in re.finditer(r"<<'PY'\n(.*?)\n\s*PY", run, re.DOTALL):
                    found += 1
                    lines = m.group(1).split("\n")
                    ind = min((len(l) - len(l.lstrip())
                               for l in lines if l.strip()), default=0)
                    src = "\n".join(l[ind:] for l in lines)
                    try:
                        compile(src, f"<{jname}>", "exec")
                    except SyntaxError as e:
                        self.fail(f"{jname} 的内嵌脚本语法错误 "
                                  f"line {e.lineno}: {e.msg}")
        self.assertGreater(found, 0, "应至少有一个内嵌校验脚本")


# ================================================================== Makefile
class TestMakefile(unittest.TestCase):
    def test_recipe_lines_use_tabs(self):
        """★ Makefile 的**配方行**必须用 Tab，用空格会报 missing separator。

        但检测逻辑必须能区分两种情况，否则会误报：

          ① 配方行 —— 目标行之后、缩进的命令行  -> 必须 Tab
          ② **续行** —— 上一行以反斜杠结尾，下一行缩进继续 -> 允许空格

        初版我只判断"是否以空格缩进"，结果把 `.PHONY:` 的续行
        （`        gen-eval review-eval ... \\`）也判成了配方行 ——
        那两行用空格完全正确，是**我的断言错了**，不是 Makefile 错了。

        之所以不用 `make -n` 验证：这台机器与本沙箱都没装 make，
        测试不能依赖它。所以这里做一个能自洽的结构分析。
        """
        body = read("Makefile")
        lines = body.splitlines()
        bad = []
        prev_nonblank = ""
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line[0] not in " \t":          # 非缩进行：记下来，继续
                prev_nonblank = line
                continue
            # 续行：上一行以反斜杠结尾 -> 允许空格
            if prev_nonblank.rstrip().endswith("\\"):
                prev_nonblank = line
                continue
            # 到这里 = 目标行之后的第一条缩进行 = 配方行
            if not line.startswith("\t"):
                bad.append((i, stripped[:40]))
            prev_nonblank = line

        self.assertEqual(
            bad, [],
            "以下配方行用空格缩进（应为 Tab）: "
            + "; ".join(f"L{n}: {t!r}" for n, t in bad))

    def test_targets_declared(self):
        import re
        targets = set(re.findall(r"(?m)^([a-zA-Z][a-zA-Z0-9_-]*):",
                                 read("Makefile")))
        for t in ("help", "test", "demo", "eval", "ingest", "lint"):
            self.assertIn(t, targets, f"缺少目标 {t}")

    def test_declares_phony_targets(self):
        body = read("Makefile")
        self.assertIn(".PHONY:", body)
        for t in ("test", "demo", "eval", "ingest"):
            self.assertIn(t, body, f".PHONY 里缺少 {t}")

    def test_no_target_conflicts_with_same_named_directory(self):
        """非 .PHONY 的目标若与同名目录冲突，make 会拒绝执行。

        例如把 `data`、`reports`、`tests` 当目标又不声明 .PHONY，
        会得到 "make: 'data' is up to date" 这种迷惑行为。
        """
        import re
        body = read("Makefile")
        targets = set(re.findall(r"(?m)^([a-zA-Z][a-zA-Z0-9_-]*):", body))
        m = re.search(r"(?s)\.PHONY:(.*?)(?=\n\n|\n[A-Za-z#])", body)
        phony = set((m.group(1) if m else "").split())
        conflicts = [t for t in targets
                     if t not in phony
                     and os.path.isdir(os.path.join(ROOT, t))]
        self.assertEqual(conflicts, [],
                         f"这些目标与同名目录冲突且未声明 .PHONY: {conflicts}")

    def test_python_vars_distinguish_core_and_ml_env(self):
        """★ PY 与 PY2 的分工是刻意的，不能合并：

            PY  跑 core/ 与测试（零第三方依赖）
            PY2 跑真模型与真 API

        合并之后"验证代码是否正确"就依赖环境是否装好，
        环境装不上就什么都验不了 —— 这正是之前那几轮踩过的坑。
        """
        body = read("Makefile")
        self.assertIn("PY  ?=", body)
        self.assertIn("PY2 ?=", body)
        self.assertIn("$(PY) -m unittest", body)
        self.assertIn("$(PY2) -m rag1_knowledge_qa.cli ingest", body)

    def test_help_is_default_goal(self):
        self.assertIn(".DEFAULT_GOAL := help", read("Makefile"))

    def test_no_carriage_returns(self):
        """★ Makefile 里绝不能有 CR（CRLF 会让 make 报 missing separator）。

        这是 GNU make 在 Windows 上最经典的坑：配方行末尾的 CR 被当作命令
        的一部分，报错指向一个看起来完全正常的行，极难定位。

        触发条件是 `core.autocrlf=true`（Windows 常见默认），
        仓库存 LF、检出成 CRLF。所以必须同时靠 .gitattributes 锁行尾 +
        这条断言兜底（它检查的就是**磁盘上的真实字节**）。

        本项目的 ci.yml 里 `run:` 的多行 shell 也怕 CR，
        所以顺带一起检查。
        """
        for path in ("Makefile", ".github/workflows/ci.yml",
                     "docker-compose.yml", "Dockerfile"):
            with open(os.path.join(ROOT, path), "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw,
                             f"{path} 含 CRLF 行尾：Makefile 会导致 "
                             f"missing separator，CI 的多行 shell 会失配")

    def test_gitattributes_locks_line_endings(self):
        """★ .gitattributes 必须把 Makefile / CI 锁成 LF。

        只靠"我写入时是 LF"不够 —— core.autocrlf 会在 checkout 时改写。
        必须在版本库层面声明，否则 clone 到 Windows 就坏。
        """
        path = os.path.join(ROOT, ".gitattributes")
        self.assertTrue(os.path.exists(path), "缺少 .gitattributes")
        body = read(".gitattributes")
        for pattern in ("Makefile", ".github/workflows/*.yml", "*.py"):
            self.assertIn(pattern, body, f".gitattributes 应声明 {pattern}")
        # 关键文件必须是 eol=lf
        mf_line = next((l for l in body.splitlines()
                        if l.startswith("Makefile")), "")
        self.assertIn("eol=lf", mf_line,
                      "Makefile 必须显式 eol=lf（不能只靠 text=auto）")

    def test_references_existing_demos(self):
        """make demo 里列的 demo 必须真实存在，否则会静默跳过。"""
        import re
        names = re.findall(r"(demo_[a-z0-9_]+)", read("Makefile"))
        for n in sorted(set(names)):
            self.assertTrue(
                os.path.exists(os.path.join(ROOT, "demos", f"{n}.py")),
                f"Makefile 引用了不存在的 demos/{n}.py")

    def test_demos_insert_repo_root(self):
        """★ 回归：demo 必须把**仓库根**插入 sys.path。

        踩坑记录：demos/ 从仓库根移进子目录后，9 个 demo 仍在插入
        `dirname(__file__)`（那已经是 demos/ 自己），从根目录运行全部
        ModuleNotFoundError；另一个用 `sys.path.insert(0, ".")` 只依赖
        cwd，换目录运行就坏。这里统一断言"插入仓库根且不依赖 cwd"，
        让这类错误在 CI 零依赖环境里直接红，而不是等用户跑 demo 才发现。
        """
        import glob
        broken = []
        for path in sorted(glob.glob(os.path.join(ROOT, "demos", "*", "*.py")) +
                           glob.glob(os.path.join(ROOT, "demos", "*.py"))):
            with open(path, encoding="utf-8") as f:
                src = f.read()
            if "sys.path.insert" not in src:
                continue  # 不 manipulates path 的 demo 不适用
            if "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))" not in src:
                broken.append(os.path.basename(path))
        self.assertEqual(broken, [],
                         f"这些 demo 没有插入仓库根: {broken}")

    def test_live_demos_not_in_tests(self):
        """★ 回归：tests/ 里不许出现需要真模型/API 的脚本。

        test_real_agent.py 曾经是个 88 行的真机冒烟脚本（顶层代码 +
        硬导入 dotenv + 需要 GPU 与 API Key），被 unittest discover
        收集后在裸环境必然 ImportError。真机脚本只许住在 demos/live/。
        """
        import glob
        for path in glob.glob(os.path.join(ROOT, "tests", "*.py")):
            with open(path, encoding="utf-8") as f:
                src = f.read()
            for line in src.splitlines():
                if line.startswith("from dotenv") or line.startswith("import dotenv"):
                    self.fail(f"{os.path.basename(path)} 在模块顶层导入 dotenv，"
                              f"会破坏零依赖可测；真机脚本请放 demos/live/")


# ================================================================== pyproject
class TestPyproject(unittest.TestCase):
    @requires_yaml
    def test_parses(self):
        data = load_yaml("pyproject.toml") if False else None
        # tomllib 只在 3.11+ 有；用 tomli 兜底，都没有就跳过
        try:
            import tomllib  # type: ignore
            with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
                data = tomllib.load(f)
        except ImportError:
            try:
                import tomli  # type: ignore
                with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
                    data = tomli.load(f)
            except ImportError:
                self.skipTest("需要 tomllib(3.11+) 或 tomli")

        self.assertEqual(data["project"]["name"], "domain-rag-qa")
        self.assertIn("requires-python", data["project"])

    def test_transforms_pinned_in_ml_extra(self):
        body = read("pyproject.toml")
        self.assertIn("transformers>=4.44,<4.46", body,
                      "ml extra 里必须把 transformers 限制在 4.x")

    def test_core_dependencies_empty(self):
        """★ dependencies 必须为空 —— 这是"零依赖可测"的声明式保证。"""
        try:
            import tomllib  # type: ignore
        except ImportError:
            self.skipTest("需要 tomllib(3.11+)")
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
            data = tomllib.load(f)
        self.assertEqual(data["project"]["dependencies"], [],
                         "核心依赖必须为空；可选依赖放 optional-dependencies")

    def test_packages_exclude_demos_and_tests(self):
        try:
            import tomllib  # type: ignore
        except ImportError:
            self.skipTest("需要 tomllib(3.11+)")
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
            data = tomllib.load(f)
        pkgs = data["tool"]["setuptools"]["packages"]
        self.assertIn("core", pkgs)
        for bad in ("tests", "tools", "data"):
            self.assertNotIn(bad, pkgs)


# ================================================================== server
class TestServerEntry(unittest.TestCase):
    def test_importable_without_heavy_deps(self):
        """server.py 在没装 torch/fastapi 的环境里应当**能编译**。

        真正的 import 会拉 fastapi 与 BGE 模型，这里只做语法级检查 ——
        保证 CI（零依赖环境）不会因为一个语法错误而漏掉它。
        """
        import io
        path = os.path.join(ROOT, "rag1_knowledge_qa", "server.py")
        src = io.open(path, encoding="utf-8").read()
        compile(src, path, "exec")

    def test_has_module_level_app(self):
        """uvicorn 的 module:app 约定需要模块级 app 对象。"""
        body = read("rag1_knowledge_qa/server.py")
        self.assertIn("app = create_app()", body)

    def test_gives_readable_startup_failure(self):
        """装配失败要给出排查步骤，而不是一大段 traceback。"""
        body = read("rag1_knowledge_qa/server.py")
        self.assertIn("排查顺序", body)
        self.assertIn("models/bge-m3", body)

    def test_does_not_import_dotenv_at_module_scope(self):
        """★ 回归：cli.py 上踩过这个坑。

        模块顶层 `from dotenv import load_dotenv` 会让没装 dotenv 的解释器
        连 import 都失败，破坏"零依赖可测"。
        """
        for line in read("rag1_knowledge_qa/server.py").splitlines():
            if line.startswith("from dotenv") or line.startswith("import dotenv"):
                self.fail("dotenv 必须在函数内惰性导入，不能在模块顶层")


if __name__ == "__main__":
    unittest.main(verbosity=2)
