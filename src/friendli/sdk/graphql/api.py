"""Code generated from api.graphql via strawberry_codegen."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import NewType

from pydantic import BaseModel, Field

from ..stub import AbstractAsyncStub, AbstractSyncStub, TypeName

UserContextGql = """
query UserContext($conn: BidirectionalConnectionInput, $sorts: ClientUserTeamSortsInput) {
  clientUser {
    teams(conn: $conn, sorts: $sorts) {
      totalCount
      edges {
        node {
          id
          name
        }
        productAccess {
          container
          dedicatedEndpoints
          serverlessEndpoints
        }
        role
        joinedAt
        default
      }
    }
  }
}
"""


AdapterPushStartGql = """
mutation AdapterPushStart($input: DedicatedModelPushAdapterStartInput!) {
  dedicatedModelPushAdapterStart(input: $input) {
    __typename
    ... on DedicatedModelPushAdapterStartSuccess {
      adapter {
        name
        id
        createdAt
      }
    }
    ... on UserPermissionError {
      message
    }
  }
}
"""


AdapterPushCompleteGql = """
mutation AdapterPushComplete($input: DedicatedModelPushAdapterCompleteInput!) {
  dedicatedModelPushAdapterComplete(input: $input) {
    __typename
    ... on UserPermissionError {
      message
    }
    ... on DedicatedModelPushAdapterCompleteSuccess {
      adapter {
        name
        id
        createdAt
      }
    }
  }
}
"""


FilePushStartGql = """
mutation FilePushStart($input: DedicatedModelPushFileStartInput!) {
  dedicatedModelPushFileStart(input: $input) {
    __typename
    ... on UserPermissionError {
      message
    }
    ... on DedicatedModelPushFileStartSuccess {
      uploadInfo {
        uploadUrl
        uploadBody
      }
    }
    ... on DedicatedModelPushFileStartAlreadyExistError {
      message
    }
  }
}
"""


FilePushCompleteGql = """
mutation FilePushComplete($input: DedicatedModelPushFileCompleteInput!) {
  dedicatedModelPushFileComplete(input: $input) {
    __typename
    ... on DedicatedModelPushFileCompleteSuccess {
      ok
    }
    ... on UserPermissionError {
      message
    }
  }
}
"""


class ClientTeamMembershipRole(str, Enum):
    OWNER = "OWNER"
    ADMIN = "ADMIN"
    BILLING = "BILLING"
    MEMBER = "MEMBER"


Base64 = NewType("Base64", bytes)


URL = NewType("URL", str)


class UserContextResultClientUserTeamsEdgesNode(BaseModel):
    id: str
    name: str | None = None


class UserContextResultClientUserTeamsEdgesProductAccess(BaseModel):
    container: bool | None = None
    dedicated_endpoints: bool | None = Field(alias="dedicatedEndpoints", default=None)
    serverless_endpoints: bool | None = Field(alias="serverlessEndpoints", default=None)


class BidirectionalConnectionInput(BaseModel):
    first: int | None = None
    skip: int
    after: Base64 | None = None
    last: int | None = None
    before: Base64 | None = None


class ClientUserTeamSortsInput(BaseModel):
    ascending: bool
    sort_by: str | None = Field(alias="sortBy", default=None)


class AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessAdapter(
    BaseModel
):
    name: str | None = None
    id: str
    created_at: datetime | None = Field(alias="createdAt", default=None)


class AdapterPushStartResultDedicatedModelPushAdapterStartUserPermissionError(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    message: str


class FileDescriptorInput(BaseModel):
    digest: str
    filename: str
    size: int


class AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccessAdapter(
    BaseModel
):
    name: str | None = None
    id: str
    created_at: datetime | None = Field(alias="createdAt", default=None)


class AdapterPushCompleteResultDedicatedModelPushAdapterCompleteUserPermissionError(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    message: str


class FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartSuccessUploadInfo(
    BaseModel
):
    upload_url: URL = Field(alias="uploadUrl")
    upload_body: dict = Field(alias="uploadBody")


class FilePushStartResultDedicatedModelPushFileStartUserPermissionError(BaseModel):
    typename__: TypeName = Field(alias="__typename")
    message: str


class FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartAlreadyExistError(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    message: str


class FilePushCompleteResultDedicatedModelPushFileCompleteDedicatedModelPushFileCompleteSuccess(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    ok: bool


class FilePushCompleteResultDedicatedModelPushFileCompleteUserPermissionError(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    message: str


class UserContextResultClientUserTeamsEdges(BaseModel):
    node: UserContextResultClientUserTeamsEdgesNode
    product_access: UserContextResultClientUserTeamsEdgesProductAccess | None = Field(
        alias="productAccess", default=None
    )
    role: ClientTeamMembershipRole | None = None
    joined_at: datetime | None = Field(alias="joinedAt", default=None)
    default: bool | None = None


class UserContextVariables(BaseModel):
    conn: BidirectionalConnectionInput | None = None
    sorts: ClientUserTeamSortsInput | None = None


class AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccess(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    adapter: AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccessAdapter


class AdapterModelCreateInput(BaseModel):
    adapter_config: FileDescriptorInput = Field(alias="adapterConfig")
    safetensors: list[FileDescriptorInput]


class DedicatedModelPushFileStartInput(BaseModel):
    model_id: str = Field(alias="modelId")
    file_input: FileDescriptorInput = Field(alias="fileInput")


class DedicatedModelPushFileCompleteInput(BaseModel):
    model_id: str = Field(alias="modelId")
    file_input: FileDescriptorInput = Field(alias="fileInput")


class AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccess(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    adapter: AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccessAdapter


class FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartSuccess(
    BaseModel
):
    typename__: TypeName = Field(alias="__typename")
    upload_info: (
        FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartSuccessUploadInfo
    ) = Field(alias="uploadInfo")


FilePushCompleteResultDedicatedModelPushFileComplete = (
    FilePushCompleteResultDedicatedModelPushFileCompleteDedicatedModelPushFileCompleteSuccess
    | FilePushCompleteResultDedicatedModelPushFileCompleteUserPermissionError
)


class UserContextResultClientUserTeams(BaseModel):
    total_count: int = Field(alias="totalCount")
    edges: list[UserContextResultClientUserTeamsEdges]


AdapterPushStartResultDedicatedModelPushAdapterStart = (
    AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccess
    | AdapterPushStartResultDedicatedModelPushAdapterStartUserPermissionError
)


class DedicatedModelPushAdapterStartInput(BaseModel):
    project_id: str = Field(alias="projectId")
    name: str | None = None
    base_model_id: str = Field(alias="baseModelId")
    model_structure: AdapterModelCreateInput = Field(alias="modelStructure")


class DedicatedModelPushAdapterCompleteInput(BaseModel):
    adapter_id: str = Field(alias="adapterId")
    model_structure: AdapterModelCreateInput = Field(alias="modelStructure")


class FilePushStartVariables(BaseModel):
    input: DedicatedModelPushFileStartInput


class FilePushCompleteVariables(BaseModel):
    input: DedicatedModelPushFileCompleteInput


AdapterPushCompleteResultDedicatedModelPushAdapterComplete = (
    AdapterPushCompleteResultDedicatedModelPushAdapterCompleteUserPermissionError
    | AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccess
)


FilePushStartResultDedicatedModelPushFileStart = (
    FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartAlreadyExistError
    | FilePushStartResultDedicatedModelPushFileStartUserPermissionError
    | FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartSuccess
)


class FilePushCompleteResult(BaseModel):
    dedicated_model_push_file_complete: (
        FilePushCompleteResultDedicatedModelPushFileComplete
    ) = Field(alias="dedicatedModelPushFileComplete")


class UserContextResultClientUser(BaseModel):
    teams: UserContextResultClientUserTeams


class AdapterPushStartResult(BaseModel):
    dedicated_model_push_adapter_start: (
        AdapterPushStartResultDedicatedModelPushAdapterStart
    ) = Field(alias="dedicatedModelPushAdapterStart")


class AdapterPushStartVariables(BaseModel):
    input: DedicatedModelPushAdapterStartInput


class AdapterPushCompleteVariables(BaseModel):
    input: DedicatedModelPushAdapterCompleteInput


class AdapterPushCompleteResult(BaseModel):
    dedicated_model_push_adapter_complete: (
        AdapterPushCompleteResultDedicatedModelPushAdapterComplete
    ) = Field(alias="dedicatedModelPushAdapterComplete")


class FilePushStartResult(BaseModel):
    dedicated_model_push_file_start: FilePushStartResultDedicatedModelPushFileStart = (
        Field(alias="dedicatedModelPushFileStart")
    )


class UserContextResult(BaseModel):
    client_user: UserContextResultClientUser | None = Field(
        alias="clientUser", default=None
    )


class BaseSyncStub(AbstractSyncStub):
    def user_context(self, variables: UserContextVariables) -> UserContextResult:
        return self._query(
            UserContextGql, response_model=UserContextResult, variables=variables
        )

    def adapter_push_start(
        self, variables: AdapterPushStartVariables
    ) -> AdapterPushStartResult:
        return self._mutation(
            AdapterPushStartGql,
            response_model=AdapterPushStartResult,
            variables=variables,
        )

    def adapter_push_complete(
        self, variables: AdapterPushCompleteVariables
    ) -> AdapterPushCompleteResult:
        return self._mutation(
            AdapterPushCompleteGql,
            response_model=AdapterPushCompleteResult,
            variables=variables,
        )

    def file_push_start(self, variables: FilePushStartVariables) -> FilePushStartResult:
        return self._mutation(
            FilePushStartGql, response_model=FilePushStartResult, variables=variables
        )

    def file_push_complete(
        self, variables: FilePushCompleteVariables
    ) -> FilePushCompleteResult:
        return self._mutation(
            FilePushCompleteGql,
            response_model=FilePushCompleteResult,
            variables=variables,
        )


class BaseAsyncStub(AbstractAsyncStub):
    async def user_context(self, variables: UserContextVariables) -> UserContextResult:
        return await self._query(
            UserContextGql, response_model=UserContextResult, variables=variables
        )

    async def adapter_push_start(
        self, variables: AdapterPushStartVariables
    ) -> AdapterPushStartResult:
        return await self._mutation(
            AdapterPushStartGql,
            response_model=AdapterPushStartResult,
            variables=variables,
        )

    async def adapter_push_complete(
        self, variables: AdapterPushCompleteVariables
    ) -> AdapterPushCompleteResult:
        return await self._mutation(
            AdapterPushCompleteGql,
            response_model=AdapterPushCompleteResult,
            variables=variables,
        )

    async def file_push_start(
        self, variables: FilePushStartVariables
    ) -> FilePushStartResult:
        return await self._mutation(
            FilePushStartGql, response_model=FilePushStartResult, variables=variables
        )

    async def file_push_complete(
        self, variables: FilePushCompleteVariables
    ) -> FilePushCompleteResult:
        return await self._mutation(
            FilePushCompleteGql,
            response_model=FilePushCompleteResult,
            variables=variables,
        )
