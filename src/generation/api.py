from time import perf_counter
from uuid import uuid4
from fastapi import APIRouter, HTTPException, status

from src.generation.excel_logger import log_request_to_excel
from src.generation.schema import (
    DBMLGenerateData,
    DBMLGenerateRequest,
    DBMLGenerateResponse,
    GenerateRequest,
    GenerateResponse,
    QuestionResult,
)
from src.generation.service import (
    build_user_prompt,
    generate_dbml,
    get_model_manager,
)

router = APIRouter(prefix="/api/v1", tags=["DBML", "Generation"])


@router.post("/generate", response_model=DBMLGenerateResponse)
def generate_dbml_endpoint(request: DBMLGenerateRequest):
    request_id = uuid4()
    started_at = perf_counter()

    user_prompt = build_user_prompt(
        enable_summary=request.enable_summary,
        summary=request.summary or "",
        user_query=request.user_query,
        dbml=request.dbml,
    )

    response_status = status.HTTP_200_OK
    error_message = None
    model_used = "custom-slm-onnx" if request.use_custom_onnx else "qwen2.5-0.5b-onnx"
    result = {}

    try:
        result = generate_dbml(
            dbml=request.dbml,
            user_query=request.user_query,
            enable_summary=request.enable_summary,
            summary=request.summary if request.enable_summary and request.summary else "",
            use_custom_onnx=request.use_custom_onnx,
        )
        model_used = result.get("model_used", model_used)

        return DBMLGenerateResponse(
            message="DBML generated successfully",
            status_code=status.HTTP_200_OK,
            data=DBMLGenerateData(
                dbml_query=result.get("dbml_query", ""),
                updated_summary=result.get("updated_summary"),
                explanation=result.get("explanation", ""),
                token_used=result.get("token_used"),
            ),
        )

    except ValueError as exc:
        response_status = status.HTTP_400_BAD_REQUEST
        error_message = str(exc)
        raise HTTPException(status_code=response_status, detail=error_message) from exc

    except HTTPException as exc:
        response_status = exc.status_code
        error_message = str(exc.detail)
        raise

    except Exception as exc:
        response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_message = str(exc)
        raise HTTPException(status_code=response_status, detail=error_message) from exc

    finally:
        total_duration = round((perf_counter() - started_at) * 1000)
        token_used = result.get("token_used", 0) or 0
        prompt_tokens = max(1, len(user_prompt) // 4)
        completion_tokens = max(0, token_used - prompt_tokens)

        log_request_to_excel(
            request_id=str(request_id),
            user_query=request.user_query,
            dbml_input=request.dbml,
            summary_enabled=request.enable_summary,
            model_used=model_used,
            status="success" if response_status < 400 else "failed",
            http_status_code=response_status,
            duration_ms=total_duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=token_used,
            dbml_output=result.get("dbml_query"),
            error_message=error_message,
        )


@router.post("/chat", response_model=GenerateResponse)
def generate_text(payload: GenerateRequest):
    manager = get_model_manager()
    questions = payload.prompt if isinstance(payload.prompt, list) else [payload.prompt]

    results: list[QuestionResult] = []
    model_name = "unknown"
    start_total_time = perf_counter()

    for q in questions:
        q_start = perf_counter()
        result_text, model_name = manager.generate(
            prompt=q,
            max_tokens=payload.max_tokens,
            use_custom_onnx=payload.use_custom_onnx,
        )
        q_duration = round(perf_counter() - q_start, 4)
        results.append(
            QuestionResult(
                question=q,
                response=result_text,
                execution_time_seconds=q_duration,
            )
        )

    total_duration = round(perf_counter() - start_total_time, 4)

    return GenerateResponse(
        model_used=model_name,
        total_execution_time_seconds=total_duration,
        results=results,
        response=results[0].response if len(results) == 1 else None,
    )
