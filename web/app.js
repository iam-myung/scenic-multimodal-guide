(() => {
  const TIMEOUT_MS = 90000;
  const BUILD_TIMEOUT_MS = 600000;
  const IS_FILE = location.protocol === "file:";

  const form = document.getElementById("askForm");
  const input = document.getElementById("queryInput");
  const askBtn = document.getElementById("askBtn");
  const clearBtn = document.getElementById("clearBtn");
  const buildBtn = document.getElementById("buildBtn");
  const statusRow = document.getElementById("statusRow");
  const errorBox = document.getElementById("errorBox");
  const result = document.getElementById("result");
  const answerText = document.getElementById("answerText");
  const sourcesList = document.getElementById("sourcesList");
  const imagePane = document.getElementById("imagePane");
  const imageEl = document.getElementById("imageEl");
  const imageCap = document.getElementById("imageCap");
  const videoPane = document.getElementById("videoPane");
  const videoWrap = document.getElementById("videoWrap");
  const videoCap = document.getElementById("videoCap");
  const fileBanner = document.getElementById("fileProtocolBanner");
  const indexBanner = document.getElementById("indexBanner");
  const indexBannerTitle = document.getElementById("indexBannerTitle");
  const indexBannerHint = document.getElementById("indexBannerHint");

  if (IS_FILE && fileBanner) {
    fileBanner.hidden = false;
  }

  /** Map knowledge_base-relative path → /kb/... */
  function kbUrlFromPath(path) {
    if (!path) return "";
    let p = String(path).replace(/\\/g, "/");
    if (p.startsWith("knowledge_base/")) {
      p = p.slice("knowledge_base/".length);
    }
    if (p.startsWith("/")) p = p.slice(1);
    return "/kb/" + p.split("/").map(encodeURIComponent).join("/");
  }

  function setStage(stage) {
    statusRow.hidden = false;
    statusRow.querySelectorAll(".pill").forEach((el) => {
      el.classList.toggle("is-active", el.dataset.stage === stage);
    });
  }

  function showError(msg) {
    errorBox.hidden = false;
    errorBox.textContent = msg;
  }

  function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function clearResult() {
    result.hidden = true;
    answerText.textContent = "";
    sourcesList.innerHTML = "";
    imagePane.hidden = true;
    imageEl.removeAttribute("src");
    imageCap.textContent = "";
    videoPane.hidden = true;
    videoWrap.innerHTML = "";
    videoCap.textContent = "";
  }

  function setIndexUi(ready, hint) {
    if (!indexBanner) return;
    indexBanner.hidden = false;
    indexBanner.classList.toggle("is-ready", !!ready);
    indexBannerTitle.textContent = ready ? "索引已就绪" : "索引未就绪";
    indexBannerHint.textContent =
      hint ||
      (ready
        ? "可以直接提问。"
        : "问答前需要先建库。可点击右侧「一键建库」（需 Embedding 按量可用）。");
    if (buildBtn) {
      buildBtn.hidden = !!ready;
      buildBtn.disabled = false;
      buildBtn.textContent = "一键建库";
    }
  }

  async function refreshStatus() {
    if (IS_FILE) return;
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (res.ok) {
        setIndexUi(!!data.index_ready, data.hint || "");
      }
    } catch {
      /* ignore — server may still be starting */
    }
  }

  async function runBuild() {
    if (IS_FILE) {
      showError("本地文件无法建库。请先启动：python -m src.cli serve --open");
      return;
    }
    clearError();
    if (buildBtn) {
      buildBtn.disabled = true;
      buildBtn.textContent = "建库中…";
    }
    setStage("thinking");
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), BUILD_TIMEOUT_MS);
    try {
      setStage("reading");
      const res = await fetch("/api/build", {
        method: "POST",
        signal: controller.signal,
      });
      let payload = null;
      try {
        payload = await res.json();
      } catch {
        payload = null;
      }
      if (!res.ok) {
        const detail =
          (payload && payload.detail) ||
          `建库失败（HTTP ${res.status}）`;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      setStage("done");
      setIndexUi(true, (payload && payload.message) || "建库成功，可以开始提问。");
      clearError();
    } catch (err) {
      statusRow.hidden = true;
      if (err && err.name === "AbortError") {
        showError("建库超时。请检查网络与 Embedding 配额后重试。");
      } else {
        showError((err && err.message) || "建库失败");
      }
      await refreshStatus();
    } finally {
      clearTimeout(timer);
      if (buildBtn) {
        buildBtn.disabled = false;
        buildBtn.textContent = "一键建库";
      }
    }
  }

  function renderSources(sources) {
    sourcesList.innerHTML = "";
    if (!Array.isArray(sources) || sources.length === 0) {
      const li = document.createElement("li");
      li.className = "source-item";
      li.textContent = "本次没有匹配到文本引用来源。";
      sourcesList.appendChild(li);
      return;
    }
    for (const s of sources) {
      const li = document.createElement("li");
      li.className = "source-item";
      const meta = document.createElement("div");
      meta.className = "source-meta";
      const sim =
        typeof s.similarity === "number" ? s.similarity.toFixed(4) : String(s.similarity);
      meta.textContent = `${s.source} · chunk ${s.chunk_id} · sim ${sim}`;
      const preview = document.createElement("div");
      preview.className = "source-preview";
      preview.textContent = s.content_preview || "";
      li.appendChild(meta);
      li.appendChild(preview);
      sourcesList.appendChild(li);
    }
  }

  function renderMedia(data) {
    imagePane.hidden = true;
    videoPane.hidden = true;
    videoWrap.innerHTML = "";

    if (data.image_ref && data.image_ref.path) {
      const url = kbUrlFromPath(data.image_ref.path);
      imageEl.src = url;
      imageEl.alt = data.image_ref.content || "相关图片";
      imageCap.textContent = data.image_ref.path;
      imagePane.hidden = false;
    }

    if (data.video_ref && data.video_ref.url) {
      const url = data.video_ref.url;
      if (/\.(mp4|webm|ogg)(\?|$)/i.test(url) || url.includes("mixkit")) {
        const video = document.createElement("video");
        video.controls = true;
        video.preload = "metadata";
        video.src = url;
        videoWrap.appendChild(video);
      } else {
        const a = document.createElement("a");
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = "在新标签打开视频";
        videoWrap.appendChild(a);
      }
      videoCap.textContent =
        (data.video_ref.description || data.video_ref.source || "") +
        (data.video_ref.url ? " · " + data.video_ref.url : "");
      videoPane.hidden = false;
    }
  }

  function renderAnswer(data) {
    // Evidence only from JSON fields — never parse answer text for sources/media.
    answerText.textContent = data.answer || "";
    renderSources(data.sources);
    renderMedia(data);
    result.hidden = false;
  }

  async function ask(query) {
    if (IS_FILE) {
      showError(
        "本地文件只能预览界面。请先启动服务：python -m src.cli serve --open"
      );
      return;
    }

    clearError();
    clearResult();
    askBtn.disabled = true;
    setStage("thinking");

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      setStage("reading");
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
        signal: controller.signal,
      });

      let payload = null;
      try {
        payload = await res.json();
      } catch {
        payload = null;
      }

      if (!res.ok) {
        const detail =
          (payload && payload.detail) ||
          (res.status === 503
            ? "索引未就绪。请先点击「一键建库」。"
            : `请求失败（HTTP ${res.status}）`);
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }

      setStage("done");
      renderAnswer(payload);
      await refreshStatus();
    } catch (err) {
      statusRow.hidden = true;
      if (err && err.name === "AbortError") {
        showError("请求超时。请确认索引已建好，且本机可访问模型服务。");
      } else {
        showError((err && err.message) || "未知错误");
      }
      await refreshStatus();
    } finally {
      clearTimeout(timer);
      askBtn.disabled = false;
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = (input.value || "").trim();
    if (!q) {
      showError("请先填写要咨询的问题。");
      return;
    }
    ask(q);
  });

  clearBtn.addEventListener("click", () => {
    input.value = "";
    clearError();
    clearResult();
    statusRow.hidden = true;
    input.focus();
  });

  if (buildBtn) {
    buildBtn.addEventListener("click", () => {
      runBuild();
    });
  }

  document.querySelectorAll(".chip[data-q]").forEach((btn) => {
    btn.addEventListener("click", () => {
      input.value = btn.getAttribute("data-q") || "";
      input.focus();
    });
  });

  refreshStatus();
})();
